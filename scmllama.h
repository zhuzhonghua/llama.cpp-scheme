#pragma once

// Binds llama.cpp into a dummyscheme VM.
//
// scmllama::Runtime owns the VM, the llama backend, and the handle table
// with RAII:
//   * construction: scm_create_vm() + llama_backend_init() + register bindings
//   * destruction:  handles.shutdown() + llama_backend_free() + scm_destroy_vm()
//
// A Runtime is not thread-safe and must stay alive as long as the VM is used.

#include "vm.h"

#include "scmllama_handle.h"

namespace scmllama {

class Runtime {
public:
  Runtime();
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

  // The dummyscheme VM owned by this runtime (null on allocation failure).
  Scheme::VM* vm() { return vm_; }

  // Handle registry for all Scheme-visible resources (model, context, ...).
  HandleTable* handles() { return &handles_; }

private:
  Scheme::VM* vm_ = nullptr;
  HandleTable handles_;
};

} // namespace scmllama
