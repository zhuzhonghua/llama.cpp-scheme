// scmllama_generate.cpp — synchronous generation loop.
//
// Scheme side:
//   (llm-generate ctx smpl prompt callback [opts]) -> generated text
//
// Options:
//   max-tokens    integer  256   stop after this many tokens
//   add-special   boolean  #t    prepend BOS when tokenizing the prompt
//   parse-special boolean  #f    treat special token markers in the prompt
//                                (e.g. <|im_start|>) as special tokens
//
// The prompt is prefilled in one decode, then each generated token is decoded
// one at a time.  The callback gets one string piece per token; an EOG token
// ends the loop.  Use (llm-reset ctx) to reuse the context for a new prompt.
//
// Returning #f from the callback stops generation: llm-generate then returns
// the text produced so far.  Any other value keeps going, so a callback that
// only appends (and returns nothing in particular) does not stop by accident.

#include "scmllama_util.h"
#include "scmllama_handle.h"

#include "llama.h"

#include <string>
#include <vector>

namespace scmllama {

namespace {

// Frees the batch on scope exit, including when a Scheme error unwinds.
struct BatchGuard
{
  llama_batch& batch;

  explicit BatchGuard(llama_batch& b) : batch(b) {}

  ~BatchGuard() { llama_batch_free(batch); }

  BatchGuard(const BatchGuard&) = delete;
  BatchGuard& operator=(const BatchGuard&) = delete;
};

} // namespace

static Scheme::ValueT scmllama_stub_llm_generate(Scheme::VM* vm,
                                                 Scheme::ValueT* ctx_value,
                                                 Scheme::ValueT* smpl_value,
                                                 Scheme::ValueT* prompt_value,
                                                 Scheme::ValueT* callback_value,
                                                 Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-generate";
  static const char* const kKeys[] = {"max-tokens", "add-special", "parse-special"};

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);

  LlmHandle* ctx_handle =
      Util::handletable(vm)->get(vm, ctx_value, kLlmContext, kWho);
  llama_context* ctx = static_cast<llama_context*>(ctx_handle->ptr);

  LlmHandle* smpl_handle =
      Util::handletable(vm)->get(vm, smpl_value, kLlmSampler, kWho);
  llama_sampler* smpl = static_cast<llama_sampler*>(smpl_handle->ptr);

  const llama_vocab* vocab = llama_model_get_vocab(llama_get_model(ctx));

  int prompt_len = 0;
  const char* prompt = Scheme::scm_get_string(vm, prompt_value, &prompt_len);

  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  int64_t max_tokens = Util::option_get_int(vm, opts, "max-tokens", 256, kWho);
  if (max_tokens < 0 || max_tokens > INT32_MAX)
    Util::error_raise(vm, "%s: max-tokens %lld is out of range", kWho,
                      static_cast<long long>(max_tokens));

  bool add_special = Util::option_get_bool(vm, opts, "add-special", true, kWho);
  bool parse_special = Util::option_get_bool(vm, opts, "parse-special", false, kWho);

  std::vector<llama_token> tokens = Util::tokenize_text(
      vm, vocab, prompt, prompt_len, add_special, parse_special, kWho);
  if (tokens.empty())
    Util::error_raise(vm, "%s: prompt produced no tokens", kWho);

  llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
  BatchGuard guard(batch);
  if (!batch.token)
    Util::error_raise(vm, "%s: failed to allocate batch", kWho);

  // Prefill the whole prompt; only the last position needs logits.
  for (size_t i = 0; i < tokens.size(); ++i)
  {
    batch.token[i] = tokens[i];
    batch.pos[i] = static_cast<llama_pos>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i == tokens.size() - 1) ? 1 : 0;
  }
  batch.n_tokens = static_cast<int32_t>(tokens.size());

  if (llama_decode(ctx, batch) != 0)
    Util::error_raise(vm, "%s: prefill decode failed", kWho);

  llama_pos n_past = static_cast<llama_pos>(tokens.size());
  std::string text;
  int32_t generated = 0;

  while (generated < max_tokens)
  {
    llama_token id = llama_sampler_sample(smpl, ctx, -1);
    if (id < 0)
      Util::error_raise(vm, "%s: sampling failed", kWho);
    if (llama_vocab_is_eog(vocab, id))
      break;

    std::vector<char> piece(256);
    int32_t n = llama_token_to_piece(vocab, id, piece.data(),
                                     static_cast<int32_t>(piece.size()), 0, true);
    if (n < 0)
    {
      piece.resize(static_cast<size_t>(-n));
      n = llama_token_to_piece(vocab, id, piece.data(),
                               static_cast<int32_t>(piece.size()), 0, true);
    }
    if (n < 0)
      Util::error_raise(vm, "%s: could not convert token %d to a piece", kWho,
                        static_cast<int>(id));

    text.append(piece.data(), static_cast<size_t>(n));

    Sgcvar2(vm, piece_value, out);
    *piece_value = Scheme::scm_make_string(vm, piece.data(), n);
    *out = Scheme::scm_make_bool(vm, false);
    if (!Scheme::scm_apply(vm, callback_value, &piece_value, 1, out))
      Util::error_raise(vm, "%s: callback failed", kWho);

    ++generated;

    // The callback returns #f to ask us to stop here; the text produced so
    // far is returned to Scheme.
    if (Scheme::scm_is_false(vm, out))
      break;

    if (generated >= max_tokens)
      break;

    // Feed the sampled token back to extend the KV cache.
    batch.n_tokens = 1;
    batch.token[0] = id;
    batch.pos[0] = n_past++;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;

    if (llama_decode(ctx, batch) != 0)
      Util::error_raise(vm, "%s: decode failed", kWho);
  }

  return Scheme::scm_make_string(vm, text.c_str(),
                                 static_cast<int>(text.size()));
}

void scmllama_register_generate_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-generate", scmllama_stub_llm_generate, true),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
