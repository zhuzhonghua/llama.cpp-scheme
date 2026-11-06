// scmllama_context.cpp — llama_context lifecycle bindings.

#include "scmllama_util.h"
#include "scmllama_handle.h"

#include "llama.h"

namespace scmllama {

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
    "embeddings", "offload-kqv", "no-perf", "op-offload",
    "swa-full", "kv-unified",
    "rope-freq-base", "rope-freq-scale",
    "yarn-ext-factor", "yarn-attn-factor",
    "yarn-beta-fast", "yarn-beta-slow", "yarn-orig-ctx"
  };

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);
  LlmHandle* model_handle = Util::model_handle(vm, value, kWho);
  const llama_model* model = Util::model_ptr(model_handle);

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
  params.offload_kqv =
      Util::option_get_bool(vm, opts, "offload-kqv", params.offload_kqv, kWho);
  params.no_perf =
      Util::option_get_bool(vm, opts, "no-perf", params.no_perf, kWho);
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
  LlmHandle* handle =
      Util::handletable(vm)->get(vm, value, kLlmContext, kWho);
  llama_context* ctx = static_cast<llama_context*>(handle->ptr);

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
  LlmHandle* handle = Util::handletable(vm)->get(vm, value, kLlmContext, kWho);
  llama_memory_clear(llama_get_memory(static_cast<llama_context*>(handle->ptr)),
                     true);
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
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
