// LoRA adapter bindings.
//
// Scheme side:
//   (llm-lora-load model path)   -> lora
//   (llm-lora-free lora)         -> #t (idempotent)
//   (llm-lora? value)            -> boolean
//   (llm-set-loras ctx loras)    -> #t
//
// `loras` is a proper list whose entries are either a lora handle (scale 1.0)
// or a (lora . scale) pair, for example:
//
//   (llm-set-loras ctx (list lora-a (cons lora-b 0.5)))
//   (llm-set-loras ctx '())          ; remove all adapters
//
// An adapter is owned by the model it was loaded for, so freeing the model
// also frees its adapters (llama_model_free deletes them).  Free adapters
// before the model when you want the memory back earlier.

#include "scmllama_util.h"
#include "scmllama_handle.h"

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace scmllama {

static void scmllama_lora_destroy(void* ptr)
{
  llama_adapter_lora_free(static_cast<llama_adapter_lora*>(ptr));
}

static Scheme::ValueT scmllama_stub_llm_lora_load(Scheme::VM* vm,
                                                  Scheme::ValueT* model_value,
                                                  Scheme::ValueT* path_value)
{
  static const char* const kWho = "llm-lora-load";

  const llama_model* model = Util::model_ptr(vm, model_value, kWho);

  int path_len = 0;
  const char* path_chars = Scheme::scm_get_string(vm, path_value, &path_len);
  std::string path(path_chars ? path_chars : "", path_chars ? path_len : 0);

  if (path.empty())
    Util::error_raise(vm, "%s: lora path is empty", kWho);

  FILE* probe = fopen(path.c_str(), "rb");
  if (!probe)
    Util::error_raise(vm, "%s: cannot open lora file '%s'", kWho, path.c_str());
  fclose(probe);

  llama_adapter_lora* lora =
      llama_adapter_lora_init(const_cast<llama_model*>(model), path.c_str());
  if (!lora)
    Util::error_raise(vm, "%s: failed to load lora '%s'", kWho, path.c_str());

  return Util::handletable(vm)->make(vm, kLlmLora, lora, scmllama_lora_destroy,
                                     const_cast<llama_model*>(model));
}

static Scheme::ValueT scmllama_stub_llm_lora_free(Scheme::VM* vm,
                                                  Scheme::ValueT* value)
{
  LlmHandle* handle =
      Util::handletable(vm)->peek(vm, value, kLlmLora, "llm-lora-free");
  Util::handletable(vm)->release(handle);
  return Scheme::scm_make_bool(vm, true);
}

static Scheme::ValueT scmllama_stub_llm_lora_p(Scheme::VM* vm,
                                               Scheme::ValueT* value)
{
  return Scheme::scm_make_bool(
      vm, Util::handletable(vm)->find(vm, value, kLlmLora) != NULL);
}

static Scheme::ValueT scmllama_stub_llm_set_loras(Scheme::VM* vm,
                                                  Scheme::ValueT* ctx_value,
                                                  Scheme::ValueT* loras_value)
{
  static const char* const kWho = "llm-set-loras";

  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  if (!Scheme::scm_is_list(vm, loras_value))
    Util::error_raise(vm, "%s: expected a list of loras, got %s", kWho,
                      Scheme::scm_typename(vm, loras_value));

  std::vector<llama_adapter_lora*> adapters;
  std::vector<float> scales;

  int index = 0;
  Scheme::ValueT* cur = loras_value;
  while (Scheme::scm_is_pair(vm, cur))
  {
    Scheme::ValueT* entry = Scheme::scm_car(vm, cur);
    Scheme::ValueT* lora_value = entry;
    float scale = 1.0f;

    if (Scheme::scm_is_pair(vm, entry))
    {
      lora_value = Scheme::scm_car(vm, entry);
      scale = Util::option_float(vm, Scheme::scm_cdr(vm, entry), 1.0f, kWho,
                                 "scale");
    }

    adapters.push_back(Util::lora_ptr(vm, lora_value, kWho));
    scales.push_back(scale);

    cur = Scheme::scm_cdr(vm, cur);
  }

  if (!Scheme::scm_is_null(vm, cur))
    Util::error_raise(vm, "%s: expected a proper list of loras", kWho);

  llama_adapter_lora** adapter_data = adapters.empty() ? NULL : adapters.data();
  float* scale_data = scales.empty() ? NULL : scales.data();
  if (llama_set_adapters_lora(ctx, adapter_data, adapters.size(), scale_data) != 0)
    Util::error_raise(vm, "%s: failed to set loras", kWho);

  return Scheme::scm_make_bool(vm, true);
}

void scmllama_register_lora_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-lora-load", scmllama_stub_llm_lora_load),
    Scheme::RegCProc("llm-lora-free", scmllama_stub_llm_lora_free),
    Scheme::RegCProc("llm-lora?",     scmllama_stub_llm_lora_p),
    Scheme::RegCProc("llm-set-loras", scmllama_stub_llm_set_loras),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
