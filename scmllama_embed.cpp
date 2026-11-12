// Embeddings bindings.
//
// Scheme side:
//   (llm-embed ctx text [opts]) -> vector of floats
//
// Options:
//   add-special   boolean  #t  prepend BOS when tokenizing the text
//   parse-special boolean  #f  treat special token markers as tokens
//
// The context must be created with (embeddings . #t).  The returned vector is:
//   pooling none (default)  last token embedding, length n-embd-out
//   pooling mean/cls/last   pooled sequence embedding, length n-embd-out
//   pooling rank            classifier output, length n-cls-out
//
// The KV cache is cleared before each call, so use a dedicated context for
// embeddings (or accept that any generation state is discarded).

#include "scmllama_util.h"
#include "scmllama_handle.h"

#include "llama.h"

#include <string>
#include <vector>

namespace scmllama {

static Scheme::ValueT scmllama_stub_llm_embed(Scheme::VM* vm,
                                              Scheme::ValueT* ctx_value,
                                              Scheme::ValueT* text_value,
                                              Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-embed";
  static const char* const kKeys[] = {"add-special", "parse-special"};

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);

  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  const bool add_special =
      Util::option_get_bool(vm, opts, "add-special", true, kWho);
  const bool parse_special =
      Util::option_get_bool(vm, opts, "parse-special", false, kWho);

  int text_len = 0;
  const char* text = Scheme::scm_get_string(vm, text_value, &text_len);

  const llama_model* model = llama_get_model(ctx);
  const llama_vocab* vocab = llama_model_get_vocab(model);

  std::vector<llama_token> tokens = Util::tokenize_text(
      vm, vocab, text, text_len, add_special, parse_special, kWho);
  if (tokens.empty())
    Util::error_raise(vm, "%s: text produced no tokens", kWho);

  llama_batch batch =
      llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
  BatchGuard guard(batch);
  if (!batch.token)
    Util::error_raise(vm, "%s: failed to allocate batch", kWho);

  for (size_t i = 0; i < tokens.size(); ++i)
  {
    batch.token[i] = tokens[i];
    batch.pos[i] = static_cast<llama_pos>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    // One output row is enough: for pooled types the whole sequence is
    // pooled, for pooling none the row is the last token's embedding.
    batch.logits[i] = (i + 1 == tokens.size()) ? 1 : 0;
  }
  batch.n_tokens = static_cast<int32_t>(tokens.size());

  // Start from a clean sequence so repeated calls do not accumulate KV cells.
  llama_memory_clear(llama_get_memory(ctx), true);

  if (llama_decode(ctx, batch) != 0)
    Util::error_raise(vm, "%s: decode failed", kWho);

  const enum llama_pooling_type pooling = llama_pooling_type(ctx);
  const float* data = NULL;
  int32_t dim = 0;

  switch (pooling)
  {
    case LLAMA_POOLING_TYPE_NONE:
      data = llama_get_embeddings_ith(ctx, -1);
      dim = llama_model_n_embd_out(model);
      break;

    case LLAMA_POOLING_TYPE_MEAN:
    case LLAMA_POOLING_TYPE_CLS:
    case LLAMA_POOLING_TYPE_LAST:
      data = llama_get_embeddings_seq(ctx, 0);
      dim = llama_model_n_embd_out(model);
      break;

    case LLAMA_POOLING_TYPE_RANK:
      data = llama_get_embeddings_seq(ctx, 0);
      dim = static_cast<int32_t>(llama_model_n_cls_out(model));
      break;

    case LLAMA_POOLING_TYPE_UNSPECIFIED:
      Util::error_raise(vm, "%s: context has no pooling type", kWho);
  }

  if (!data)
    Util::error_raise(vm,
                      "%s: context returned no embedding; create it with "
                      "(embeddings . #t)",
                      kWho);
  if (dim <= 0)
    Util::error_raise(vm, "%s: invalid embedding dimension %d", kWho,
                      static_cast<int>(dim));

  Sgcvar1(vm, result);
  *result = Scheme::scm_make_vector(vm, dim);
  for (int32_t i = 0; i < dim; ++i)
    Scheme::scm_vector_set(vm, result, static_cast<int>(i),
                           Scheme::scm_make_float(vm, data[i]));

  return *result;
}

void scmllama_register_embed_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-embed", scmllama_stub_llm_embed, true),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
