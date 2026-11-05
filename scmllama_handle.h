#pragma once

// HandleTable keeps a registry of every Scheme-visible resource (model,
// context, sampler, batch).  Each handle is a lightuserdata wrapping an
// LlmHandle record; records are never freed before shutdown so stale Scheme
// values are detected instead of dereferencing freed memory.
//
#include "vm.h"

#include <vector>

namespace scmllama {

// Kind tag stored in every handle.  It keeps a model from being used where a
// context/sampler/batch is expected.
enum LlmKind {
  kLlmModel = 1,
  kLlmContext,
  kLlmSampler,
  kLlmBatch
};

// Scheme sees a lightuserdata pointing at an LlmHandle.
struct LlmHandle {
  LlmKind kind;
  void* ptr;
  bool alive;
  void (*destroy)(void*);
};

class HandleTable {
public:
  HandleTable() {}
  HandleTable(const HandleTable&) = delete;
  HandleTable& operator=(const HandleTable&) = delete;

  // Registers a new live handle and returns its lightuserdata value.
  Scheme::ValueT make(Scheme::VM* vm, LlmKind kind, void* ptr,
                      void (*destroy)(void*));

  // Like get(), but a dead handle is returned as-is (for idempotent free).
  LlmHandle* peek(Scheme::VM* vm, Scheme::ValueT* value, LlmKind kind,
                  const char* who);

  // Returns a live handle of the expected kind; raises a Scheme error otherwise.
  LlmHandle* get(Scheme::VM* vm, Scheme::ValueT* value, LlmKind kind,
                 const char* who);

  // Non-throwing lookup: NULL unless `value` is a live handle of `kind`.
  LlmHandle* find(Scheme::VM* vm, Scheme::ValueT* value, LlmKind kind);

  // Marks a handle as freed and releases the underlying object once.
  void release(LlmHandle* handle);

  // Releases every handle still alive.  Called by Runtime before backend shutdown.
  void shutdown();

private:
  bool issued(const void* ptr) const;
  static const char* kind_name(LlmKind kind);

  std::vector<LlmHandle*> handles_;
};

} // namespace scmllama
