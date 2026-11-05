// scmllama_batch.cpp — llama_batch bindings.

#include "scmllama_util.h"
#include "scmllama_handle.h"

#include "llama.h"

#include <vector>

namespace scmllama {

// ---------------------------------------------------------------------------
// batch helpers

static void scmllama_batch_destroy(void* ptr)
{
  if (ptr)
    llama_batch_free(*static_cast<llama_batch*>(ptr));
  delete static_cast<llama_batch*>(ptr);
}

// Build a batch from a list/vector of token ids (seq 0, logits for last token).
static Scheme::ValueT scmllama_stub_llm_batch_from_tokens(Scheme::VM* vm,
                                                          Scheme::ValueT* value,
                                                          Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-batch-from-tokens";
  static const char* const kKeys[] = {"last-logits"};

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);

  Util::option_require_known(
      vm, opts, kKeys, static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])),
      kWho);

  bool last_logits = Util::option_get_bool(vm, opts, "last-logits", true, kWho);

  std::vector<llama_token> tokens = Util::token_list(vm, value, kWho);
  if (tokens.empty())
    Util::error_raise(vm, "%s: token list must not be empty", kWho);

  llama_batch* batch = new llama_batch;
  *batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
  if (!batch->token)
  {
    delete batch;
    Util::error_raise(vm, "%s: failed to allocate batch", kWho);
  }

  // llama_batch_init only allocates capacity; n_tokens starts at 0.
  batch->n_tokens = static_cast<int32_t>(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++)
  {
    batch->token[i]     = tokens[i];
    batch->pos[i]       = static_cast<llama_pos>(i);
    batch->n_seq_id[i]  = 1;
    batch->seq_id[i][0] = 0;
    batch->logits[i]    = last_logits && (i == tokens.size() - 1) ? 1 : 0;
  }

  return Util::handletable(vm)->make(vm, kLlmBatch, batch, scmllama_batch_destroy);
}

// Set logits flag for a token index.
static Scheme::ValueT scmllama_stub_llm_batch_set_logits(Scheme::VM* vm,
                                                         Scheme::ValueT* value,
                                                         Scheme::ValueT* index,
                                                         Scheme::ValueT* on)
{
  static const char* const kWho = "llm-batch-set-logits";

  LlmHandle* handle = Util::handletable(vm)->get(vm, value, kLlmBatch, kWho);
  llama_batch* batch = static_cast<llama_batch*>(handle->ptr);

  int64_t idx = Scheme::scm_get_int(vm, index);
  if (idx < 0 || idx >= batch->n_tokens)
    Util::error_raise(vm, "%s: index %d out of range [0, %d)", kWho, (int)idx, batch->n_tokens);

  batch->logits[idx] = Scheme::scm_get_bool(vm, on) ? 1 : 0;
  return Scheme::scm_make_bool(vm, true);
}

// Return the number of tokens in the batch.
static Scheme::ValueT scmllama_stub_llm_batch_n_tokens(Scheme::VM* vm,
                                                       Scheme::ValueT* value)
{
  static const char* const kWho = "llm-batch-n-tokens";
  LlmHandle* handle = Util::handletable(vm)->get(vm, value, kLlmBatch, kWho);
  llama_batch* batch = static_cast<llama_batch*>(handle->ptr);
  return Scheme::scm_make_int(vm, batch->n_tokens);
}

// Free the batch.
static Scheme::ValueT scmllama_stub_llm_batch_free(Scheme::VM* vm,
                                                   Scheme::ValueT* value)
{
  LlmHandle* handle = Util::handletable(vm)->peek(vm, value, kLlmBatch,
                                                  "llm-batch-free");
  Util::handletable(vm)->release(handle);
  return Scheme::scm_make_bool(vm, true);
}

static Scheme::ValueT scmllama_stub_llm_batch_p(Scheme::VM* vm,
                                                Scheme::ValueT* value)
{
  return Scheme::scm_make_bool(
      vm, Util::handletable(vm)->find(vm, value, kLlmBatch) != NULL);
}

// ---------------------------------------------------------------------------
// decode

// Run llama_decode on the batch. Returns #t on success, raises on error.
static Scheme::ValueT scmllama_stub_llm_decode(Scheme::VM* vm,
                                               Scheme::ValueT* ctx_value,
                                               Scheme::ValueT* batch_value)
{
  static const char* const kWho = "llm-decode";

  LlmHandle* ctx_handle =
      Util::handletable(vm)->get(vm, ctx_value, kLlmContext, kWho);
  llama_context* ctx = static_cast<llama_context*>(ctx_handle->ptr);

  LlmHandle* batch_handle =
      Util::handletable(vm)->get(vm, batch_value, kLlmBatch, kWho);
  llama_batch* batch = static_cast<llama_batch*>(batch_handle->ptr);

  int32_t rc = llama_decode(ctx, *batch);
  if (rc != 0)
    Util::error_raise(vm, "%s: llama_decode failed with code %d", kWho, rc);

  return Scheme::scm_make_bool(vm, true);
}

// ---------------------------------------------------------------------------
// registration

void scmllama_register_batch_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-batch-from-tokens", scmllama_stub_llm_batch_from_tokens, true),
    Scheme::RegCProc("llm-batch-set-logits", scmllama_stub_llm_batch_set_logits),
    Scheme::RegCProc("llm-batch-n-tokens",   scmllama_stub_llm_batch_n_tokens),
    Scheme::RegCProc("llm-batch-free",       scmllama_stub_llm_batch_free),
    Scheme::RegCProc("llm-batch?",           scmllama_stub_llm_batch_p),
    Scheme::RegCProc("llm-decode",           scmllama_stub_llm_decode),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
