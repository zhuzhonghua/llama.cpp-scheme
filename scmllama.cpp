// Minimal binding: exposes a few harmless libllama functions to Scheme.
//
// Scheme side:
//   (llm-version)      -> string   llama_version()
//   (llm-system-info)  -> string   llama_print_system_info()
//
// The llama backend lifecycle is managed by scmllama::Runtime (RAII):
// constructor calls llama_backend_init(), destructor calls llama_backend_free().

#include "scmllama.h"

#include "llama.h"

namespace scmllama {

static Scheme::ValueT scmllama_stub_llm_version(Scheme::VM* vm)
{
  return Scheme::scm_make_string(vm, llama_version());
}

static Scheme::ValueT scmllama_stub_llm_system_info(Scheme::VM* vm)
{
  const char* info = llama_print_system_info();
  return Scheme::scm_make_string(vm, info ? info : "");
}

// Registers all Scheme bindings. The table is a function-local static, so it
// has no external linkage.
static void scmllama_register_lib(Scheme::VM* vm)
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
  scmllama_register_lib(vm_);
}

Runtime::~Runtime()
{
  llama_backend_free();
  Scheme::scm_destroy_vm(vm_);
}

} // namespace scmllama
