// Model lifecycle and metadata bindings.
//
// Scheme side:
//   (llm-model-load path [opts])  -> model handle
//   (llm-model-free model)        -> #t (idempotent)
//   (llm-model? value)            -> boolean
//   (llm-model-info model)        -> hash-table

#include "scmllama.h"
#include "scmllama_util.h"

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace scmllama {

static void scmllama_model_destroy(void* ptr)
{
  llama_model_free(static_cast<llama_model*>(ptr));
}

static std::string scmllama_model_desc(const llama_model* model)
{
  std::vector<char> buf(256);
  int32_t n = llama_model_desc(model, buf.data(), buf.size());
  if (n < 0)
    return std::string();
  if (static_cast<size_t>(n) >= buf.size())
  {
    buf.resize(n + 1);
    n = llama_model_desc(model, buf.data(), buf.size());
    if (n < 0)
      return std::string();
  }
  return std::string(buf.data(), n);
}

static void scmllama_put_int(Scheme::VM* vm, Scheme::ValueT* table,
                             const char* key, int64_t value)
{
  Sgcvar2(vm, keyv, valv);
  *keyv = Scheme::scm_make_string(vm, key);
  *valv = Scheme::scm_make_int(vm, value);
  Scheme::scm_table_set(vm, table, keyv, *valv);
}

static void scmllama_put_bool(Scheme::VM* vm, Scheme::ValueT* table,
                              const char* key, bool value)
{
  Sgcvar2(vm, keyv, valv);
  *keyv = Scheme::scm_make_string(vm, key);
  *valv = Scheme::scm_make_bool(vm, value);
  Scheme::scm_table_set(vm, table, keyv, *valv);
}

static void scmllama_put_string(Scheme::VM* vm, Scheme::ValueT* table,
                                const char* key, const std::string& value)
{
  Sgcvar2(vm, keyv, valv);
  *keyv = Scheme::scm_make_string(vm, key);
  *valv = Scheme::scm_make_string(vm, value.data(), value.size());
  Scheme::scm_table_set(vm, table, keyv, *valv);
}

static Scheme::ValueT scmllama_stub_llm_model_load(
    Scheme::VM* vm, Scheme::ValueT* path_value, Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-model-load";
  static const char* const kKeys[] = {
    "n-gpu-layers",
    "main-gpu",
    "vocab-only",
    "check-tensors",
    "no-alloc",
    "no-host",
    "use-extra-bufts",
    "load-mtp"
  };

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);

  int path_len = 0;
  const char* path_chars = Scheme::scm_get_string(vm, path_value, &path_len);
  std::string path(path_chars ? path_chars : "", path_chars ? path_len : 0);

  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  llama_model_params params = llama_model_default_params();

  int64_t n_gpu_layers =
      Util::option_get_int(vm, opts, "n-gpu-layers", params.n_gpu_layers, kWho);
  if (n_gpu_layers < -1 || n_gpu_layers > 100000)
    Util::error_raise(vm, "%s: n-gpu-layers %lld is out of range", kWho,
                 static_cast<long long>(n_gpu_layers));
  params.n_gpu_layers = static_cast<int32_t>(n_gpu_layers);

  int64_t main_gpu =
      Util::option_get_int(vm, opts, "main-gpu", params.main_gpu, kWho);
  if (main_gpu < 0 || main_gpu > 1000)
    Util::error_raise(vm, "%s: main-gpu %lld is out of range", kWho,
                 static_cast<long long>(main_gpu));
  params.main_gpu = static_cast<int32_t>(main_gpu);

  params.vocab_only =
      Util::option_get_bool(vm, opts, "vocab-only", params.vocab_only, kWho);
  params.check_tensors =
      Util::option_get_bool(vm, opts, "check-tensors", params.check_tensors, kWho);
  params.no_alloc =
      Util::option_get_bool(vm, opts, "no-alloc", params.no_alloc, kWho);
  params.no_host =
      Util::option_get_bool(vm, opts, "no-host", params.no_host, kWho);
  params.use_extra_bufts = Util::option_get_bool(vm, opts, "use-extra-bufts",
                                                 params.use_extra_bufts, kWho);
  params.load_mtp =
      Util::option_get_bool(vm, opts, "load-mtp", params.load_mtp, kWho);

  if (path.empty())
    Util::error_raise(vm, "%s: model path is empty", kWho);

  FILE* probe = fopen(path.c_str(), "rb");
  if (!probe)
    Util::error_raise(vm, "%s: cannot open model file '%s'", kWho, path.c_str());
  fclose(probe);

  llama_model* model = llama_model_load_from_file(path.c_str(), params);
  if (!model)
    Util::error_raise(vm, "%s: failed to load model '%s'", kWho, path.c_str());

  return Util::handletable(vm)->make(vm, kLlmModel, model,
                                     scmllama_model_destroy);
}

static Scheme::ValueT scmllama_stub_llm_model_free(Scheme::VM* vm,
                                                   Scheme::ValueT* value)
{
  LlmHandle* handle = Util::handletable(vm)->peek(vm, value, kLlmModel, "llm-model-free");
  Util::handletable(vm)->release(handle);
  return Scheme::scm_make_bool(vm, true);
}

static Scheme::ValueT scmllama_stub_llm_model_p(Scheme::VM* vm,
                                                Scheme::ValueT* value)
{
  return Scheme::scm_make_bool(vm, Util::handletable(vm)->find(vm, value, kLlmModel) != NULL);
}

static Scheme::ValueT scmllama_stub_llm_model_info(Scheme::VM* vm,
                                                   Scheme::ValueT* value)
{
  const llama_model* model = Util::model_ptr(vm, value, "llm-model-info");
  const llama_vocab* vocab = Util::model_vocab(vm, value, "llm-model-info");

  Sgcvar1(vm, table);
  *table = Scheme::scm_make_table(vm);

  scmllama_put_string(vm, table, "desc", scmllama_model_desc(model));
  scmllama_put_int(vm, table, "ftype",
                   static_cast<int64_t>(llama_model_ftype(model)));
  scmllama_put_int(vm, table, "n-params",
                   static_cast<int64_t>(llama_model_n_params(model)));
  scmllama_put_int(vm, table, "size",
                   static_cast<int64_t>(llama_model_size(model)));
  scmllama_put_int(vm, table, "n-ctx-train",
                   llama_model_n_ctx_train(model));
  scmllama_put_int(vm, table, "n-embd", llama_model_n_embd(model));
  scmllama_put_int(vm, table, "n-layer", llama_model_n_layer(model));
  scmllama_put_int(vm, table, "n-head", llama_model_n_head(model));
  scmllama_put_int(vm, table, "n-head-kv", llama_model_n_head_kv(model));
  scmllama_put_int(vm, table, "n-vocab", llama_vocab_n_tokens(vocab));
  scmllama_put_int(vm, table, "bos-token", llama_vocab_bos(vocab));
  scmllama_put_int(vm, table, "eos-token", llama_vocab_eos(vocab));
  scmllama_put_bool(vm, table, "has-encoder",
                    llama_model_has_encoder(model));
  scmllama_put_bool(vm, table, "has-decoder",
                    llama_model_has_decoder(model));

  const char* chat_template = llama_model_chat_template(model, NULL);
  scmllama_put_string(vm, table, "chat-template",
                      chat_template ? chat_template : "");

  return *table;
}

void scmllama_register_model_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-model-load", scmllama_stub_llm_model_load, true),
    Scheme::RegCProc("llm-model-free", scmllama_stub_llm_model_free),
    Scheme::RegCProc("llm-model?",     scmllama_stub_llm_model_p),
    Scheme::RegCProc("llm-model-info", scmllama_stub_llm_model_info),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
