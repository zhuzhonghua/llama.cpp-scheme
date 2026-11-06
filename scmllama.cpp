// Minimal binding: exposes a few harmless libllama functions to Scheme.
//
// Scheme side:
//   (llm-version)      -> string   llama_version()
//   (llm-system-info)  -> string   llama_print_system_info()
//
// The llama backend lifecycle is managed by scmllama::Runtime (RAII):
// constructor calls llama_backend_init(), destructor calls llama_backend_free().

#include "scmllama.h"

#include "scmllama_util.h"

#include "llama.h"

namespace scmllama {

// Binding registration (defined in scmllama_model.cpp / scmllama_vocab.cpp /
// scmllama_context.cpp / scmllama_batch.cpp / scmllama_sampler.cpp /
// scmllama_generate.cpp).
extern void scmllama_register_model_lib(Scheme::VM* vm);
extern void scmllama_register_vocab_lib(Scheme::VM* vm);
extern void scmllama_register_context_lib(Scheme::VM* vm);
extern void scmllama_register_batch_lib(Scheme::VM* vm);
extern void scmllama_register_sampler_lib(Scheme::VM* vm);
extern void scmllama_register_generate_lib(Scheme::VM* vm);

const char* const Runtime::kRegistryKey = "~scmllama.runtime";

static Scheme::ValueT scmllama_stub_llm_version(Scheme::VM* vm)
{
  return Scheme::scm_make_string(vm, llama_version());
}

static Scheme::ValueT scmllama_stub_llm_system_info(Scheme::VM* vm)
{
  const char* info = llama_print_system_info();
  return Scheme::scm_make_string(vm, info ? info : "");
}

static void scmllama_register_basic_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-version",     scmllama_stub_llm_version),
    Scheme::RegCProc("llm-system-info", scmllama_stub_llm_system_info),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

Runtime::Runtime() : vm_(Scheme::scm_create_vm())
{
  llama_backend_init();
  if (vm_)
  {
    Scheme::scm_registry_set(vm_, kRegistryKey,
                             Scheme::scm_make_lightuserdata(vm_, this));
    scmllama_register_basic_lib(vm_);
    scmllama_register_model_lib(vm_);
    scmllama_register_vocab_lib(vm_);
    scmllama_register_context_lib(vm_);
    scmllama_register_batch_lib(vm_);
    scmllama_register_sampler_lib(vm_);
    scmllama_register_generate_lib(vm_);
  }
}

Runtime::~Runtime()
{
  if (vm_)
    Scheme::scm_registry_unset(vm_, kRegistryKey);
  handles_.shutdown();
  llama_backend_free();
  if (vm_)
    Scheme::scm_destroy_vm(vm_);
}

} // namespace scmllama
