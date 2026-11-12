// scmllama_generate.cpp — synchronous generation loop.
//
// Scheme side:
//   (llm-generate ctx smpl prompt callback [opts])           -> generated text
//   (llm-chat-generate ctx smpl messages callback [opts])    -> generated text
//
// llm-generate options:
//   max-tokens    integer  256   stop after this many tokens
//   add-special   boolean  #t    prepend BOS when tokenizing the prompt
//   parse-special boolean  #f    treat special token markers in the prompt
//                                (e.g. <|im_start|>) as special tokens
//   keep-kv       boolean  #f    append the prompt to the existing KV cache
//                                instead of starting from position 0.  Use for
//                                multi-turn dialogue: the new prompt is placed
//                                right after the tokens already cached, so the
//                                model sees the full conversation.
//
// llm-chat-generate options:
//   template      string   model's built-in template
//   add-assistant boolean  #t    append the assistant turn header
//   max-tokens    integer  256
//   add-special   boolean  #f    the rendered template already contains BOS
//   parse-special boolean  #t    <|im_start|>-style markers are real tokens
//
// llm-chat-generate renders `messages` (a proper list of (role . content)
// pairs) with the model's chat template, then generates from that prompt.
// Because the whole conversation is re-rendered on every call, the KV cache is
// cleared first and the context can be reused across conversations; keep-kv is
// therefore not supported, since appending the full history to an existing KV
// cache would duplicate it.  For incremental turns, call
// llm-chat-template-apply on the new turn and pass only that text to
// llm-generate with (keep-kv . #t).
//
// The prompt is prefilled in one decode, then each generated token is decoded
// one at a time.  The callback gets one string piece per token; an EOG token
// ends the loop.  Use (llm-reset ctx) to reuse the context for a new prompt.
//
// Returning #f from the callback stops generation: the function then returns
// the text produced so far.  Any other value keeps going, so a callback that
// only appends (and returns nothing in particular) does not stop by accident.

#include "scmllama_util.h"

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scmllama {

// Shared generation loop used by llm-generate and llm-chat-generate.
// `prompt` does not need to be NUL terminated; `prompt_len` is its length.
static Scheme::ValueT scmllama_generate_impl(
    Scheme::VM* vm, llama_context* ctx, llama_sampler* smpl, const char* prompt,
    int prompt_len, Scheme::ValueT* callback_value, int64_t max_tokens,
    bool add_special, bool parse_special, bool keep_kv, const char* kWho)
{
  const llama_vocab* vocab = llama_model_get_vocab(llama_get_model(ctx));

  std::vector<llama_token> tokens = Util::tokenize_text(
      vm, vocab, prompt, prompt_len, add_special, parse_special, kWho);
  if (tokens.empty())
    Util::error_raise(vm, "%s: prompt produced no tokens", kWho);

  llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
  BatchGuard guard(batch);
  if (!batch.token)
    Util::error_raise(vm, "%s: failed to allocate batch", kWho);

  // Prefill the whole prompt; only the last position needs logits.
  // With keep-kv, the prompt is appended right after the tokens already in the
  // KV cache so the model sees the full conversation history.
  llama_pos base_pos = 0;
  if (keep_kv)
    base_pos = static_cast<llama_pos>(Util::kv_seq_used(ctx, 0));

  for (size_t i = 0; i < tokens.size(); ++i)
  {
    batch.token[i] = tokens[i];
    batch.pos[i] = base_pos + static_cast<llama_pos>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i == tokens.size() - 1) ? 1 : 0;
  }
  batch.n_tokens = static_cast<int32_t>(tokens.size());

  if (llama_decode(ctx, batch) != 0)
    Util::error_raise(vm, "%s: prefill decode failed", kWho);

  llama_pos n_past = base_pos + static_cast<llama_pos>(tokens.size());
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

static int64_t scmllama_generate_max_tokens(Scheme::VM* vm,
                                            Scheme::ValueT* opts,
                                            const char* who)
{
  int64_t max_tokens = Util::option_get_int(vm, opts, "max-tokens", 256, who);
  if (max_tokens < 0 || max_tokens > INT32_MAX)
    Util::error_raise(vm, "%s: max-tokens %lld is out of range", who,
                      static_cast<long long>(max_tokens));
  return max_tokens;
}

// Handles plus options shared by both entry points.  They differ only in how
// the prompt is obtained, their default tokenization flags, and whether
// keep-kv is allowed.
struct ScmllamaGenerateSetup {
  Scheme::ValueT* opts;
  llama_context* ctx;
  llama_sampler* smpl;
  int64_t max_tokens;
  bool add_special;
  bool parse_special;
  bool keep_kv;
};

static ScmllamaGenerateSetup scmllama_generate_setup(
    Scheme::VM* vm, Scheme::ValueT* ctx_value, Scheme::ValueT* smpl_value,
    Scheme::ValueT* rest, const char* who, const char* const* keys,
    int key_count, bool default_add_special, bool default_parse_special)
{
  ScmllamaGenerateSetup setup;
  setup.opts = Util::option_from_rest(vm, rest, who);
  setup.ctx = Util::context_ptr(vm, ctx_value, who);
  setup.smpl = Util::sampler_ptr(vm, smpl_value, who);

  Util::option_require_known(vm, setup.opts, keys, key_count, who);

  setup.max_tokens = scmllama_generate_max_tokens(vm, setup.opts, who);
  setup.add_special = Util::option_get_bool(
      vm, setup.opts, "add-special", default_add_special, who);
  setup.parse_special = Util::option_get_bool(
      vm, setup.opts, "parse-special", default_parse_special, who);
  setup.keep_kv =
      Util::option_get_bool(vm, setup.opts, "keep-kv", false, who);
  return setup;
}

static Scheme::ValueT scmllama_stub_llm_generate(Scheme::VM* vm,
                                                 Scheme::ValueT* ctx_value,
                                                 Scheme::ValueT* smpl_value,
                                                 Scheme::ValueT* prompt_value,
                                                 Scheme::ValueT* callback_value,
                                                 Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-generate";
  static const char* const kKeys[] = {"max-tokens", "add-special",
                                      "parse-special", "keep-kv"};

  const ScmllamaGenerateSetup setup = scmllama_generate_setup(
      vm, ctx_value, smpl_value, rest, kWho, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), true, false);

  int prompt_len = 0;
  const char* prompt = Scheme::scm_get_string(vm, prompt_value, &prompt_len);

  return scmllama_generate_impl(vm, setup.ctx, setup.smpl, prompt, prompt_len,
                                callback_value, setup.max_tokens,
                                setup.add_special, setup.parse_special,
                                setup.keep_kv, kWho);
}

static Scheme::ValueT scmllama_stub_llm_chat_generate(
    Scheme::VM* vm, Scheme::ValueT* ctx_value, Scheme::ValueT* smpl_value,
    Scheme::ValueT* messages_value, Scheme::ValueT* callback_value,
    Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-chat-generate";
  static const char* const kKeys[] = {"template", "add-assistant", "max-tokens",
                                      "add-special", "parse-special", "keep-kv"};

  const ScmllamaGenerateSetup setup = scmllama_generate_setup(
      vm, ctx_value, smpl_value, rest, kWho, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), false, true);

  // The full conversation is re-rendered below, so appending it to an
  // existing KV cache would duplicate the history.  Incremental callers
  // should use llm-chat-template-apply + llm-generate instead.
  if (setup.keep_kv)
    Util::error_raise(
        vm,
        "%s: keep-kv is not supported because the whole conversation is "
        "re-rendered on every call; use llm-chat-template-apply + "
        "llm-generate with (keep-kv . #t) for incremental turns",
        kWho);

  const std::string prompt = Util::chat_render(
      vm, llama_get_model(setup.ctx), messages_value, setup.opts, kWho);
  if (prompt.size() > static_cast<size_t>(INT32_MAX))
    Util::error_raise(vm, "%s: rendered prompt is too long", kWho);

  // Every call carries the full conversation, so it always starts a fresh
  // sequence.  Clearing here lets callers reuse a context across conversations
  // without an explicit (llm-reset ctx).  This is why keep-kv is rejected.
  llama_memory_clear(llama_get_memory(setup.ctx), true);

  return scmllama_generate_impl(vm, setup.ctx, setup.smpl, prompt.data(),
                                static_cast<int>(prompt.size()), callback_value,
                                setup.max_tokens, setup.add_special,
                                setup.parse_special, false, kWho);
}

void scmllama_register_generate_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-generate", scmllama_stub_llm_generate, true),
    Scheme::RegCProc("llm-chat-generate", scmllama_stub_llm_chat_generate, true),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
