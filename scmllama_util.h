#pragma once

// Shared infrastructure for the llama.cpp bindings: stateless helpers grouped
// by prefix inside the single `Util` class.  Stateful HandleTable lives in
// scmllama_handle.h (it is owned by Runtime).

#include "vm.h"

struct llama_model;
struct llama_vocab;

namespace scmllama {

struct LlmHandle;  // defined in scmllama_handle.h

// Stateless helpers.  Methods are grouped by prefix:
//   error_*   error reporting
//   model_*   handle -> llama object accessors
class Util {
public:
  // Raises a Scheme error with a formatted message.
  [[noreturn]] static void error_raise(Scheme::VM* vm, const char* fmt, ...);

  static const llama_model* model_ptr(LlmHandle* handle);
  static const llama_vocab* model_vocab(LlmHandle* handle);
};

} // namespace scmllama
