#pragma once

// Binds llama.cpp into a dummyscheme VM.
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

public:
  // The registry key used to look up this Runtime from bindings (vm -> Runtime*).
  // Must outlive the Runtime itself (string literal storage).
  static const char* const kRegistryKey;

private:
  Scheme::VM* vm_ = nullptr;
  HandleTable handles_;
};

} // namespace scmllama
