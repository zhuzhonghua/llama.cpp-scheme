// scmllama_batch.cpp — llama_batch bindings.
//
// Scheme side:
//   (llm-batch-from-tokens tokens [opts])     -> batch (seq 0, positions 0..n-1)
//   (llm-batch-new capacity [opts])           -> empty batch, filled token by token
//   (llm-batch-add-token batch token pos [opts]) -> batch index of the new token
//   (llm-batch-clear batch)                   -> #t (n_tokens := 0, keep capacity)
//   (llm-batch-set-logits batch index on?)    -> #t
//   (llm-batch-n-tokens batch)                -> integer
//   (llm-batch-free batch)                    -> #t (idempotent)
//   (llm-batch? value)                        -> boolean
//   (llm-decode ctx batch)                    -> #t
//
// llm-batch-new options:
//   n-seq-max     integer  1    max seq ids a single token may carry
//
// llm-batch-add-token options:
//   seq-id        integer  0    sequence the token belongs to
//   seq-ids       list     none explicit seq id list (one batch token shared by
//                               several sequences); mutually exclusive with seq-id
//   logits        boolean  #f   produce logits for this token (needed to sample
//                               from it later)
//
// Batches built with llm-batch-new support multi-sequence decoding: several
// sequences can be prefilled or extended in a single llama_decode call, with
// per-token positions.  The caller owns the positions and must keep them
// consecutive within each sequence (the KV cache never overwrites).
//
// llm-decode validates the batch before calling llama.cpp: empty batches,
// tokens without a seq id, seq ids outside the context's n-seq-max and
// negative positions are reported as Scheme errors instead of aborting.

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
  {
    BatchBox* box = static_cast<BatchBox*>(ptr);
    llama_batch_free(box->batch);
    delete box;
  }
}

// Allocates a batch of `capacity` tokens and wraps it in a Scheme handle.
static Scheme::ValueT scmllama_batch_make(Scheme::VM* vm, int32_t capacity,
                                          int32_t n_seq_max, const char* who)
{
  BatchBox* box = new BatchBox;
  box->batch = llama_batch_init(capacity, 0, n_seq_max);
  box->capacity = capacity;
  box->n_seq_max = n_seq_max;

  if (!box->batch.token)
  {
    llama_batch_free(box->batch);
    delete box;
    Util::error_raise(vm, "%s: failed to allocate batch", who);
  }

  return Util::handletable(vm)->make(vm, kLlmBatch, box, scmllama_batch_destroy);
}

// Parses the seq-id / seq-ids options into the per-token seq id list.
static std::vector<llama_seq_id> scmllama_batch_seq_ids(
    Scheme::VM* vm, Scheme::ValueT* opts, int32_t n_seq_max, const char* who)
{
  std::vector<llama_seq_id> seq_ids;

  Scheme::ValueT* list_slot = Util::option_find(vm, opts, "seq-ids", who);
  Scheme::ValueT* one_slot = Util::option_find(vm, opts, "seq-id", who);

  if (list_slot && one_slot)
    Util::error_raise(vm, "%s: use either seq-id or seq-ids, not both", who);

  if (!list_slot)
  {
    int64_t id = Util::option_get_int(vm, opts, "seq-id", 0, who);
    if (id < 0 || id >= n_seq_max)
      Util::error_raise(vm, "%s: seq id %lld is out of range [0, %d)", who,
                        static_cast<long long>(id), n_seq_max);
    seq_ids.push_back(static_cast<llama_seq_id>(id));
    return seq_ids;
  }

  if (!Scheme::scm_is_list(vm, list_slot))
    Util::error_raise(vm, "%s: option 'seq-ids' expects a list, got %s", who,
                      Scheme::scm_typename(vm, list_slot));

  Scheme::ValueT* cur = list_slot;
  while (Scheme::scm_is_pair(vm, cur))
  {
    Scheme::ValueT* item = Scheme::scm_car(vm, cur);
    if (!Scheme::scm_is_int(vm, item))
      Util::error_raise(vm, "%s: option 'seq-ids' expects integers, got %s",
                        who, Scheme::scm_typename(vm, item));

    int64_t id = static_cast<int64_t>(Scheme::scm_get_int(vm, item));
    if (id < 0 || id >= n_seq_max)
      Util::error_raise(vm, "%s: seq id %lld is out of range [0, %d)", who,
                        static_cast<long long>(id), n_seq_max);
    seq_ids.push_back(static_cast<llama_seq_id>(id));

    cur = Scheme::scm_cdr(vm, cur);
  }
  if (!Scheme::scm_is_null(vm, cur))
    Util::error_raise(vm, "%s: option 'seq-ids' expects a proper list", who);
  if (seq_ids.empty())
    Util::error_raise(vm, "%s: option 'seq-ids' must not be empty", who);

  // Each token owns n_seq_max seq id slots; never write past them.
  if (seq_ids.size() > static_cast<size_t>(n_seq_max))
    Util::error_raise(vm, "%s: at most %d seq ids per token are allowed", who,
                      n_seq_max);

  return seq_ids;
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

  Scheme::ValueT handle = scmllama_batch_make(
      vm, static_cast<int32_t>(tokens.size()), 1, kWho);
  BatchBox* box = Util::batch_box(vm, &handle, kWho);
  llama_batch* batch = &box->batch;

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

  return handle;
}

// Create an empty batch with room for `capacity` tokens.
static Scheme::ValueT scmllama_stub_llm_batch_new(Scheme::VM* vm,
                                                  Scheme::ValueT* capacity_value,
                                                  Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-batch-new";
  static const char* const kKeys[] = {"n-seq-max"};

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);

  Util::option_require_known(
      vm, opts, kKeys, static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])),
      kWho);

  if (!Scheme::scm_is_int(vm, capacity_value))
    Util::error_raise(vm, "%s: expected a capacity (integer), got %s", kWho,
                      Scheme::scm_typename(vm, capacity_value));

  int64_t capacity = static_cast<int64_t>(Scheme::scm_get_int(vm, capacity_value));
  if (capacity < 1 || capacity > INT32_MAX)
    Util::error_raise(vm, "%s: capacity %lld is out of range", kWho,
                      static_cast<long long>(capacity));

  int64_t n_seq_max = Util::option_get_int(vm, opts, "n-seq-max", 1, kWho);
  if (n_seq_max < 1 || n_seq_max > INT32_MAX)
    Util::error_raise(vm, "%s: n-seq-max %lld is out of range", kWho,
                      static_cast<long long>(n_seq_max));

  return scmllama_batch_make(vm, static_cast<int32_t>(capacity),
                             static_cast<int32_t>(n_seq_max), kWho);
}

// Append one token at an explicit position / sequence.
static Scheme::ValueT scmllama_stub_llm_batch_add_token(
    Scheme::VM* vm, Scheme::ValueT* value, Scheme::ValueT* token_value,
    Scheme::ValueT* pos_value, Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-batch-add-token";
  static const char* const kKeys[] = {"seq-id", "seq-ids", "logits"};

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);
  BatchBox* box = Util::batch_box(vm, value, kWho);

  Util::option_require_known(
      vm, opts, kKeys, static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])),
      kWho);

  llama_token token = Util::token_from_value(vm, token_value, kWho);
  bool logits = Util::option_get_bool(vm, opts, "logits", false, kWho);

  if (!Scheme::scm_is_int(vm, pos_value))
    Util::error_raise(vm, "%s: expected a position (integer), got %s", kWho,
                      Scheme::scm_typename(vm, pos_value));
  int64_t pos = static_cast<int64_t>(Scheme::scm_get_int(vm, pos_value));
  if (pos < 0 || pos > INT32_MAX)
    Util::error_raise(vm, "%s: position %lld is out of range", kWho,
                      static_cast<long long>(pos));

  const std::vector<llama_seq_id> seq_ids =
      scmllama_batch_seq_ids(vm, opts, box->n_seq_max, kWho);

  llama_batch* batch = &box->batch;
  if (batch->n_tokens >= box->capacity)
    Util::error_raise(vm,
                      "%s: batch is full (capacity %d); build a larger one "
                      "with llm-batch-new or call llm-batch-clear",
                      kWho, box->capacity);

  const int32_t index = batch->n_tokens;

  batch->token[index] = token;
  batch->pos[index] = static_cast<llama_pos>(pos);
  batch->n_seq_id[index] = static_cast<int32_t>(seq_ids.size());
  for (size_t j = 0; j < seq_ids.size(); j++)
    batch->seq_id[index][j] = seq_ids[j];
  batch->logits[index] = logits ? 1 : 0;

  batch->n_tokens = index + 1;

  // The index identifies this token in the batch; pass it to
  // llm-sampler-sample to sample that token's logits.
  return Scheme::scm_make_int(vm, index);
}

// Drop the tokens but keep the allocated arrays (for per-step loops).
static Scheme::ValueT scmllama_stub_llm_batch_clear(Scheme::VM* vm,
                                                    Scheme::ValueT* value)
{
  static const char* const kWho = "llm-batch-clear";
  Util::batch_ptr(vm, value, kWho)->n_tokens = 0;
  return Scheme::scm_make_bool(vm, true);
}

// Set logits flag for a token index.
static Scheme::ValueT scmllama_stub_llm_batch_set_logits(Scheme::VM* vm,
                                                         Scheme::ValueT* value,
                                                         Scheme::ValueT* index,
                                                         Scheme::ValueT* on)
{
  static const char* const kWho = "llm-batch-set-logits";

  llama_batch* batch = Util::batch_ptr(vm, value, kWho);

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
  llama_batch* batch = Util::batch_ptr(vm, value, kWho);
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

// Reject batches that llama_decode would abort on (empty, missing/unknown seq
// ids, negative positions) with a readable Scheme error instead.
static void scmllama_batch_validate(Scheme::VM* vm, llama_context* ctx,
                                    const llama_batch* batch, const char* who)
{
  if (batch->n_tokens <= 0)
    Util::error_raise(vm, "%s: batch is empty", who);

  const int32_t n_seq_max = static_cast<int32_t>(llama_n_seq_max(ctx));

  for (int32_t i = 0; i < batch->n_tokens; i++)
  {
    if (batch->pos[i] < 0)
      Util::error_raise(vm, "%s: token %d has a negative position (%d)", who, i,
                        static_cast<int>(batch->pos[i]));

    if (batch->n_seq_id[i] < 1)
      Util::error_raise(vm, "%s: token %d has no seq id", who, i);

    for (int32_t j = 0; j < batch->n_seq_id[i]; j++)
    {
      const llama_seq_id seq_id = batch->seq_id[i][j];
      if (seq_id < 0 || seq_id >= n_seq_max)
        Util::error_raise(vm,
                          "%s: token %d uses seq id %d, but the context was "
                          "created with n-seq-max %d",
                          who, i, static_cast<int>(seq_id), n_seq_max);
    }
  }
}

// Run llama_decode on the batch. Returns #t on success, raises on error.
static Scheme::ValueT scmllama_stub_llm_decode(Scheme::VM* vm,
                                               Scheme::ValueT* ctx_value,
                                               Scheme::ValueT* batch_value)
{
  static const char* const kWho = "llm-decode";

  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  llama_batch* batch = Util::batch_ptr(vm, batch_value, kWho);

  scmllama_batch_validate(vm, ctx, batch, kWho);

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
    Scheme::RegCProc("llm-batch-new",         scmllama_stub_llm_batch_new, true),
    Scheme::RegCProc("llm-batch-add-token",   scmllama_stub_llm_batch_add_token, true),
    Scheme::RegCProc("llm-batch-clear",       scmllama_stub_llm_batch_clear),
    Scheme::RegCProc("llm-batch-set-logits",  scmllama_stub_llm_batch_set_logits),
    Scheme::RegCProc("llm-batch-n-tokens",    scmllama_stub_llm_batch_n_tokens),
    Scheme::RegCProc("llm-batch-free",        scmllama_stub_llm_batch_free),
    Scheme::RegCProc("llm-batch?",            scmllama_stub_llm_batch_p),
    Scheme::RegCProc("llm-decode",            scmllama_stub_llm_decode),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
