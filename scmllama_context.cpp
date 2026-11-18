// scmllama_context.cpp — llama_context lifecycle bindings.

#include "scmllama_util.h"
#include "scmllama_handle.h"

#include "llama.h"

namespace scmllama {

// Parses a pooling-type option value: symbol or string, e.g. 'mean or "cls".
static enum llama_pooling_type scmllama_pooling_type(Scheme::VM* vm,
                                                    Scheme::ValueT* slot,
                                                    const char* who)
{
  const char* chars = NULL;
  int len = 0;

  if (Scheme::scm_is_symbol(vm, slot))
    chars = Scheme::scm_tostring(vm, slot, &len);
  else if (Scheme::scm_is_string(vm, slot))
    chars = Scheme::scm_get_string(vm, slot, &len);
  else
    Util::error_raise(vm,
                      "%s: option 'pooling-type' expects a symbol or string, "
                      "got %s",
                      who, Scheme::scm_typename(vm, slot));

  const std::string value(chars ? chars : "", chars ? len : 0);

  if (value == "none") return LLAMA_POOLING_TYPE_NONE;
  if (value == "mean") return LLAMA_POOLING_TYPE_MEAN;
  if (value == "cls")  return LLAMA_POOLING_TYPE_CLS;
  if (value == "last") return LLAMA_POOLING_TYPE_LAST;
  if (value == "rank") return LLAMA_POOLING_TYPE_RANK;

  Util::error_raise(vm,
                    "%s: unknown pooling-type '%s' "
                    "(expected none/mean/cls/last/rank)",
                    who, value.c_str());
}

static void scmllama_context_destroy(void* ptr)
{
  if (ptr)
    llama_free(static_cast<llama_context*>(ptr));
}

static Scheme::ValueT scmllama_stub_llm_context_new(Scheme::VM* vm,
                                                    Scheme::ValueT* value,
                                                    Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-context-new";
  static const char* const kKeys[] = {
    "n-ctx", "n-batch", "n-ubatch", "n-seq-max",
    "n-threads", "n-threads-batch",
    "embeddings", "pooling-type", "offload-kqv", "no-perf", "op-offload",
    "swa-full", "kv-unified",
    "rope-freq-base", "rope-freq-scale",
    "yarn-ext-factor", "yarn-attn-factor",
    "yarn-beta-fast", "yarn-beta-slow", "yarn-orig-ctx"
  };

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);
  const llama_model* model = Util::model_ptr(vm, value, kWho);

  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  llama_context_params params = llama_context_default_params();

  params.n_ctx = static_cast<uint32_t>(
      Util::option_get_int(vm, opts, "n-ctx", params.n_ctx, kWho));
  params.n_batch = static_cast<uint32_t>(
      Util::option_get_int(vm, opts, "n-batch", params.n_batch, kWho));
  params.n_ubatch = static_cast<uint32_t>(
      Util::option_get_int(vm, opts, "n-ubatch", params.n_ubatch, kWho));
  params.n_seq_max = static_cast<uint32_t>(
      Util::option_get_int(vm, opts, "n-seq-max", params.n_seq_max, kWho));
  params.n_threads = static_cast<int32_t>(
      Util::option_get_int(vm, opts, "n-threads", params.n_threads, kWho));
  params.n_threads_batch = static_cast<int32_t>(
      Util::option_get_int(vm, opts, "n-threads-batch", params.n_threads_batch, kWho));

  params.embeddings =
      Util::option_get_bool(vm, opts, "embeddings", params.embeddings, kWho);

  Scheme::ValueT* pooling_slot =
      Util::option_find(vm, opts, "pooling-type", kWho);
  if (pooling_slot)
    params.pooling_type = scmllama_pooling_type(vm, pooling_slot, kWho);

  params.offload_kqv =
      Util::option_get_bool(vm, opts, "offload-kqv", params.offload_kqv, kWho);
  params.no_perf =
      Util::option_get_bool(vm, opts, "no-perf", false, kWho);
  params.op_offload =
      Util::option_get_bool(vm, opts, "op-offload", params.op_offload, kWho);
  params.swa_full =
      Util::option_get_bool(vm, opts, "swa-full", params.swa_full, kWho);
  params.kv_unified =
      Util::option_get_bool(vm, opts, "kv-unified", params.kv_unified, kWho);

  // float options
  params.rope_freq_base = Util::option_get_float(
      vm, opts, "rope-freq-base", params.rope_freq_base, kWho);
  params.rope_freq_scale = Util::option_get_float(
      vm, opts, "rope-freq-scale", params.rope_freq_scale, kWho);
  params.yarn_ext_factor = Util::option_get_float(
      vm, opts, "yarn-ext-factor", params.yarn_ext_factor, kWho);
  params.yarn_attn_factor = Util::option_get_float(
      vm, opts, "yarn-attn-factor", params.yarn_attn_factor, kWho);
  params.yarn_beta_fast = Util::option_get_float(
      vm, opts, "yarn-beta-fast", params.yarn_beta_fast, kWho);
  params.yarn_beta_slow = Util::option_get_float(
      vm, opts, "yarn-beta-slow", params.yarn_beta_slow, kWho);

  params.yarn_orig_ctx = static_cast<uint32_t>(
      Util::option_get_int(vm, opts, "yarn-orig-ctx", params.yarn_orig_ctx, kWho));

  llama_context* ctx = llama_init_from_model(
      const_cast<llama_model*>(model), params);
  if (!ctx)
    Util::error_raise(vm, "%s: failed to create context", kWho);

  return Util::handletable(vm)->make(vm, kLlmContext, ctx,
                                     scmllama_context_destroy);
}

static Scheme::ValueT scmllama_stub_llm_context_free(Scheme::VM* vm,
                                                     Scheme::ValueT* value)
{
  LlmHandle* handle = Util::handletable(vm)->peek(vm, value, kLlmContext,
                                                  "llm-context-free");
  Util::handletable(vm)->release(handle);
  return Scheme::scm_make_bool(vm, true);
}

static Scheme::ValueT scmllama_stub_llm_context_p(Scheme::VM* vm,
                                                  Scheme::ValueT* value)
{
  return Scheme::scm_make_bool(
      vm, Util::handletable(vm)->find(vm, value, kLlmContext) != NULL);
}

static Scheme::ValueT scmllama_stub_llm_context_info(Scheme::VM* vm,
                                                     Scheme::ValueT* value)
{
  static const char* const kWho = "llm-context-info";
  llama_context* ctx = Util::context_ptr(vm, value, kWho);

  Sgcvar1(vm, table);
  *table = Scheme::scm_make_table(vm);

  Sgcvar2(vm, keyv, valv);

  *keyv = Scheme::scm_make_string(vm, "n-ctx", 5);
  *valv = Scheme::scm_make_int(vm, llama_n_ctx(ctx));
  Scheme::scm_table_set(vm, table, keyv, *valv);

  *keyv = Scheme::scm_make_string(vm, "n-batch", 7);
  *valv = Scheme::scm_make_int(vm, llama_n_batch(ctx));
  Scheme::scm_table_set(vm, table, keyv, *valv);

  *keyv = Scheme::scm_make_string(vm, "n-seq-max", 9);
  *valv = Scheme::scm_make_int(vm, llama_n_seq_max(ctx));
  Scheme::scm_table_set(vm, table, keyv, *valv);

  *keyv = Scheme::scm_make_string(vm, "n-threads", 9);
  *valv = Scheme::scm_make_int(vm, 0);  // TODO: no getter for n_threads
  Scheme::scm_table_set(vm, table, keyv, *valv);

  *keyv = Scheme::scm_make_string(vm, "embeddings", 10);
  *valv = Scheme::scm_make_bool(vm, false);  // TODO: no getter for embeddings
  Scheme::scm_table_set(vm, table, keyv, *valv);

  return *table;
}

// Drop the KV cache so the same context can take a fresh prompt.
static Scheme::ValueT scmllama_stub_llm_reset(Scheme::VM* vm,
                                              Scheme::ValueT* value)
{
  static const char* const kWho = "llm-reset";
  llama_context* ctx = Util::context_ptr(vm, value, kWho);
  llama_memory_clear(llama_get_memory(ctx), true);
  return Scheme::scm_make_bool(vm, true);
}

// Number of tokens currently held in the KV cache for a sequence.  Optional
// seq (default 0) selects which sequence: (llm-kv-pos ctx) or
// (llm-kv-pos ctx seq).  Historically this only ever queried seq 0 (for
// llm-generate's keep-kv option), so the default keeps that behaviour.
// Returns 0 when the sequence has no KV.
static Scheme::ValueT scmllama_stub_llm_kv_pos(Scheme::VM* vm,
                                                Scheme::ValueT* value,
                                                Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-kv-pos";
  llama_context* ctx = Util::context_ptr(vm, value, kWho);

  llama_seq_id seq = 0;
  if (rest && !Scheme::scm_is_null(vm, rest))
  {
    const int n = Scheme::scm_proper_list_len(vm, rest);
    if (n != 1)
      Util::error_raise(vm, "%s: expected at most one seq id", kWho);

    Scheme::ValueT* seq_value = Scheme::scm_car(vm, rest);
    if (!Scheme::scm_is_int(vm, seq_value))
      Util::error_raise(vm, "%s: seq id expects an integer, got %s", kWho,
                        Scheme::scm_typename(vm, seq_value));

    const int64_t s = static_cast<int64_t>(Scheme::scm_get_int(vm, seq_value));
    const int32_t n_seq_max = static_cast<int32_t>(llama_n_seq_max(ctx));
    if (s < 0 || s >= n_seq_max)
      Util::error_raise(vm, "%s: seq id out of range [0,%d): got %lld", kWho,
                        n_seq_max, static_cast<long long>(s));
    seq = static_cast<llama_seq_id>(s);
  }

  return Scheme::scm_make_int(vm, Util::kv_seq_used(ctx, seq));
}

// Remove KV cache cells of sequence 0 at positions [p0, p1).
// Pass p1 = -1 to remove everything from p0 onwards (tail truncation,
// e.g. "forget the last N tokens" before retrying a generation).
// Returns #t if any cell was actually removed.
// Note: removing a middle range leaves a hole; llama.cpp does not shift the
// remaining positions down.  Prefer truncating the tail.
static Scheme::ValueT scmllama_stub_llm_kv_remove(Scheme::VM* vm,
                                                  Scheme::ValueT* ctx_value,
                                                  Scheme::ValueT* p0_value,
                                                  Scheme::ValueT* p1_value)
{
  static const char* const kWho = "llm-kv-remove";
  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  const llama_pos p0 =
      static_cast<llama_pos>(Scheme::scm_get_int(vm, p0_value));
  const llama_pos p1 =
      static_cast<llama_pos>(Scheme::scm_get_int(vm, p1_value));

  const bool removed = llama_memory_seq_rm(llama_get_memory(ctx), 0, p0, p1);
  return Scheme::scm_make_bool(vm, removed);
}

// Shift the positions of sequence 0 cells in [p0, p1) by `delta`.
// p1 = -1 means "from p0 onwards".  Used to close the hole left by
// llm-kv-remove: after removing the oldest n cells at the front, call
// (llm-kv-shift ctx 0 -1 -n) so the remaining positions start at 0 again
// and keep-kv can append right after them.  This is what llama-cli calls
// "context shift"; the KV data itself is not recomputed.
static Scheme::ValueT scmllama_stub_llm_kv_shift(Scheme::VM* vm,
                                                 Scheme::ValueT* ctx_value,
                                                 Scheme::ValueT* p0_value,
                                                 Scheme::ValueT* p1_value,
                                                 Scheme::ValueT* delta_value)
{
  static const char* const kWho = "llm-kv-shift";
  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  const llama_pos p0 =
      static_cast<llama_pos>(Scheme::scm_get_int(vm, p0_value));
  const llama_pos p1 =
      static_cast<llama_pos>(Scheme::scm_get_int(vm, p1_value));
  const llama_pos delta =
      static_cast<llama_pos>(Scheme::scm_get_int(vm, delta_value));

  llama_memory_seq_add(llama_get_memory(ctx), 0, p0, p1, delta);
  return Scheme::scm_make_bool(vm, true);
}

// Copy KV cells of sequence `src` in [p0, p1) into sequence `dst`.  This is the
// complement of llm-kv-remove/shift for multi-sequence contexts: both sequences
// share the same cached prefix and can then diverge from the same position
// (beam expand / fork).  With no optional positions the whole sequence is copied.
// p0 = -1 means [0, p1), p1 = -1 means [p0, inf), matching the C API.
// Optional args are passed positionally: (llm-kv-copy ctx src dst) or
// (llm-kv-copy ctx src dst p0 p1).
static Scheme::ValueT scmllama_stub_llm_kv_copy(Scheme::VM* vm,
                                                 Scheme::ValueT* ctx_value,
                                                 Scheme::ValueT* src_value,
                                                 Scheme::ValueT* dst_value,
                                                 Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-kv-copy";
  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  const int src = static_cast<int>(Scheme::scm_get_int(vm, src_value));
  const int dst = static_cast<int>(Scheme::scm_get_int(vm, dst_value));

  llama_pos p0 = -1;
  llama_pos p1 = -1;
  if (Scheme::scm_is_pair(vm, rest)) {
    p0 = static_cast<llama_pos>(Scheme::scm_get_int(vm, Scheme::scm_car(vm, rest)));
    Scheme::ValueT* second = Scheme::scm_cdr(vm, rest);
    if (!Scheme::scm_is_pair(vm, second))
      Util::error_raise(vm, "%s: p0 and p1 must be given together", kWho);
    p1 = static_cast<llama_pos>(Scheme::scm_get_int(vm, Scheme::scm_car(vm, second)));
    if (Scheme::scm_is_pair(vm, Scheme::scm_cdr(vm, second)))
      Util::error_raise(vm, "%s: too many optional positions (expected p0 p1)",
                        kWho);
  }

  const int32_t n_seq_max = static_cast<int32_t>(llama_n_seq_max(ctx));
  if (src < 0 || src >= n_seq_max || dst < 0 || dst >= n_seq_max)
    Util::error_raise(vm,
                      "%s: seq id out of range [0,%d): got src=%d dst=%d",
                      kWho, n_seq_max, src, dst);
  if (p0 < -1 || p1 < -1)
    Util::error_raise(vm,
                      "%s: positions must be >= -1 (-1 = open end), "
                      "got p0=%d p1=%d",
                      kWho, p0, p1);

  llama_memory_seq_cp(llama_get_memory(ctx), src, dst, p0, p1);
  return Scheme::scm_make_bool(vm, true);
}

// Keep KV cells of only `seq`, dropping every other sequence (beam prune).
// Pairs with llm-kv-copy: fork a prefix into several sequences, then keep the
// winner.  Returns #t.
static Scheme::ValueT scmllama_stub_llm_kv_keep(Scheme::VM* vm,
                                                 Scheme::ValueT* ctx_value,
                                                 Scheme::ValueT* seq_value)
{
  static const char* const kWho = "llm-kv-keep";
  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  const int seq = static_cast<int>(Scheme::scm_get_int(vm, seq_value));
  const int32_t n_seq_max = static_cast<int32_t>(llama_n_seq_max(ctx));
  if (seq < 0 || seq >= n_seq_max)
    Util::error_raise(vm, "%s: seq id out of range [0,%d): got %d",
                      kWho, n_seq_max, seq);

  llama_memory_seq_keep(llama_get_memory(ctx), seq);
  return Scheme::scm_make_bool(vm, true);
}

// Put an integer value into the perf hash-table under the given string key.
static void scmllama_perf_puti(Scheme::VM* vm, Scheme::ValueT* table,
                               Scheme::ValueT* keyv, Scheme::ValueT* valv,
                               const char* key, int32_t v)
{
  *keyv = Scheme::scm_make_string(vm, key, (int)strlen(key));
  *valv = Scheme::scm_make_int(vm, v);
  Scheme::scm_table_set(vm, table, keyv, *valv);
}

// Put a float value into the perf hash-table under the given string key.
static void scmllama_perf_putf(Scheme::VM* vm, Scheme::ValueT* table,
                               Scheme::ValueT* keyv, Scheme::ValueT* valv,
                               const char* key, double v)
{
  *keyv = Scheme::scm_make_string(vm, key, (int)strlen(key));
  *valv = Scheme::scm_make_float(vm, v);
  Scheme::scm_table_set(vm, table, keyv, *valv);
}

// Performance counters for the context (prefill + decode timing/tokens).
static Scheme::ValueT scmllama_stub_llm_perf(Scheme::VM* vm,
                                              Scheme::ValueT* value)
{
  static const char* const kWho = "llm-perf";
  llama_context* ctx = Util::context_ptr(vm, value, kWho);

  const llama_perf_context_data perf = llama_perf_context(ctx);

  const double prefill_tps =
      perf.t_p_eval_ms > 0.0 ? perf.n_p_eval / (perf.t_p_eval_ms / 1000.0) : 0.0;
  const double decode_tps =
      perf.t_eval_ms > 0.0 ? perf.n_eval / (perf.t_eval_ms / 1000.0) : 0.0;

  Sgcvar1(vm, table);
  *table = Scheme::scm_make_table(vm);

  Sgcvar2(vm, keyv, valv);

  scmllama_perf_puti(vm, table, keyv, valv, "n-prefill", perf.n_p_eval);
  scmllama_perf_puti(vm, table, keyv, valv, "n-decode", perf.n_eval);
  scmllama_perf_putf(vm, table, keyv, valv, "t-prefill-ms", perf.t_p_eval_ms);
  scmllama_perf_putf(vm, table, keyv, valv, "t-decode-ms", perf.t_eval_ms);
  scmllama_perf_putf(vm, table, keyv, valv, "t-load-ms", perf.t_load_ms);
  scmllama_perf_putf(vm, table, keyv, valv, "prefill-tps", prefill_tps);
  scmllama_perf_putf(vm, table, keyv, valv, "decode-tps", decode_tps);

  return *table;
}

// Reset the context's performance counters.
static Scheme::ValueT scmllama_stub_llm_perf_reset(Scheme::VM* vm,
                                                    Scheme::ValueT* value)
{
  static const char* const kWho = "llm-perf-reset";
  llama_perf_context_reset(Util::context_ptr(vm, value, kWho));
  return Scheme::scm_make_bool(vm, true);
}

// Save the context state (KV cache + logits) to a session file.
// Token history is managed by the Scheme side, so an empty token list is stored.
static Scheme::ValueT scmllama_stub_llm_state_save(Scheme::VM* vm,
                                                    Scheme::ValueT* ctx_value,
                                                    Scheme::ValueT* path_value)
{
  static const char* const kWho = "llm-state-save";
  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  int path_len = 0;
  const char* path_chars = Scheme::scm_get_string(vm, path_value, &path_len);
  std::string path(path_chars ? path_chars : "", path_chars ? path_len : 0);

  if (!llama_state_save_file(ctx, path.c_str(), NULL, 0))
    Util::error_raise(vm, "%s: failed to save state to '%s'", kWho, path.c_str());

  return Scheme::scm_make_bool(vm, true);
}

// Load a session file back into the context (restores KV cache + logits).
static Scheme::ValueT scmllama_stub_llm_state_load(Scheme::VM* vm,
                                                    Scheme::ValueT* ctx_value,
                                                    Scheme::ValueT* path_value)
{
  static const char* const kWho = "llm-state-load";
  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  int path_len = 0;
  const char* path_chars = Scheme::scm_get_string(vm, path_value, &path_len);
  std::string path(path_chars ? path_chars : "", path_chars ? path_len : 0);

  size_t n_token_count = 0;
  if (!llama_state_load_file(ctx, path.c_str(), NULL, 0, &n_token_count))
    Util::error_raise(vm, "%s: failed to load state from '%s'", kWho, path.c_str());

  return Scheme::scm_make_bool(vm, true);
}

void scmllama_register_context_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-context-new",  scmllama_stub_llm_context_new, true),
    Scheme::RegCProc("llm-context-free", scmllama_stub_llm_context_free),
    Scheme::RegCProc("llm-context?",     scmllama_stub_llm_context_p),
    Scheme::RegCProc("llm-context-info", scmllama_stub_llm_context_info),
    Scheme::RegCProc("llm-reset",        scmllama_stub_llm_reset),
    Scheme::RegCProc("llm-kv-pos",       scmllama_stub_llm_kv_pos, true),
    Scheme::RegCProc("llm-kv-remove",    scmllama_stub_llm_kv_remove),
    Scheme::RegCProc("llm-kv-shift",     scmllama_stub_llm_kv_shift),
    Scheme::RegCProc("llm-kv-copy",      scmllama_stub_llm_kv_copy, true),
    Scheme::RegCProc("llm-kv-keep",      scmllama_stub_llm_kv_keep),
    Scheme::RegCProc("llm-perf",         scmllama_stub_llm_perf),
    Scheme::RegCProc("llm-perf-reset",   scmllama_stub_llm_perf_reset),
    Scheme::RegCProc("llm-state-save",   scmllama_stub_llm_state_save),
    Scheme::RegCProc("llm-state-load",   scmllama_stub_llm_state_load),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
