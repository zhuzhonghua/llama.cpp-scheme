#pragma once

// Shared infrastructure for the llama.cpp bindings: stateless helpers grouped
// by prefix inside the single `Util` class.  Stateful HandleTable lives in
// scmllama_handle.h (owned by Runtime, reachable via Util::runtime(vm)).

#include "vm.h"

#include <cstdint>

struct llama_model;
struct llama_vocab;

namespace scmllama {

struct LlmHandle;    // defined in scmllama_handle.h
class HandleTable;   // defined in scmllama_handle.h
class Runtime;       // defined in scmllama.h

// Stateless helpers.  Methods are grouped by prefix:
//   error_*   error reporting
//   option_*  alist option access
//   runtime   recover the Runtime from a VM
//   handle_*  handle lookup shortcuts (one call = get + kind check)
//   model_*   handle -> llama object accessors
class Util {
public:
  // Raises a Scheme error with a formatted message.
  [[noreturn]] static void error_raise(Scheme::VM* vm, const char* fmt, ...);

  // --- registry ---

  // Recovers the Runtime that owns `vm` (or nullptr if none registered).
  static Runtime* runtime(Scheme::VM* vm);
  static HandleTable* handletable(Scheme::VM* vm);

  // --- handle shortcuts ---

  static LlmHandle* model_handle(Scheme::VM* vm, Scheme::ValueT* value,
                                 const char* who);

  // --- model accessors ---

  static const llama_model* model_ptr(LlmHandle* handle);
  static const llama_vocab* model_vocab(LlmHandle* handle);

  // --- option parsing ---

  // Extracts the optional options alist from a rest-argument list: either no
  // options or exactly one alist argument.  Returns NULL when absent.
  static Scheme::ValueT* option_from_rest(Scheme::VM* vm, Scheme::ValueT* rest,
                                          const char* who);

  // Value slot for `key`, or NULL when absent.
  static Scheme::ValueT* option_find(Scheme::VM* vm, Scheme::ValueT* alist,
                                     const char* key, const char* who);

  // Raises when the alist contains a key outside `keys`.
  static void option_require_known(Scheme::VM* vm, Scheme::ValueT* alist,
                                   const char* const* keys, int count,
                                   const char* who);

  static bool option_bool(Scheme::VM* vm, Scheme::ValueT* slot, bool def,
                          const char* who, const char* key);
  static int64_t option_int(Scheme::VM* vm, Scheme::ValueT* slot, int64_t def,
                            const char* who, const char* key);

  // option_find + option_bool / option_int in one call: returns `def` when
  // `key` is absent from `alist`.
  static bool option_get_bool(Scheme::VM* vm, Scheme::ValueT* alist,
                              const char* key, bool def, const char* who);
  static int64_t option_get_int(Scheme::VM* vm, Scheme::ValueT* alist,
                                const char* key, int64_t def, const char* who);
};

} // namespace scmllama
