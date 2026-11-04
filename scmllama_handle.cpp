#include "scmllama_handle.h"

#include "scmllama_util.h"

namespace scmllama {

bool HandleTable::issued(const void* ptr) const
{
  for (size_t i = 0; i < handles_.size(); ++i)
  {
    if (handles_[i] == ptr)
      return true;
  }
  return false;
}

const char* HandleTable::kind_name(LlmKind kind)
{
  switch (kind)
  {
    case kLlmModel:   return "model";
    case kLlmContext: return "context";
    case kLlmSampler: return "sampler";
    case kLlmBatch:   return "batch";
  }
  return "handle";
}

Scheme::ValueT HandleTable::make(Scheme::VM* vm, LlmKind kind, void* ptr,
                                 void (*destroy)(void*))
{
  LlmHandle* handle = new LlmHandle;
  handle->kind = kind;
  handle->ptr = ptr;
  handle->alive = true;
  handle->destroy = destroy;
  handles_.push_back(handle);
  return Scheme::scm_make_lightuserdata(vm, handle);
}

LlmHandle* HandleTable::peek(Scheme::VM* vm, Scheme::ValueT* value,
                             LlmKind kind, const char* who)
{
  if (!value || !Scheme::scm_is_lightuserdata(vm, value))
  {
    Util::error_raise(vm, "%s: expected a %s handle, got %s", who, kind_name(kind),
                      value ? Scheme::scm_typename(vm, value) : "nothing");
  }

  void* raw = Scheme::scm_to_lightuserdata(vm, value);
  if (!issued(raw))
    Util::error_raise(vm, "%s: expected a %s handle", who, kind_name(kind));

  LlmHandle* handle = static_cast<LlmHandle*>(raw);
  if (handle->kind != kind)
  {
    Util::error_raise(vm, "%s: expected a %s handle, got a %s handle", who,
                      kind_name(kind), kind_name(handle->kind));
  }
  return handle;
}

LlmHandle* HandleTable::get(Scheme::VM* vm, Scheme::ValueT* value,
                            LlmKind kind, const char* who)
{
  LlmHandle* handle = peek(vm, value, kind, who);
  if (!handle->alive)
    Util::error_raise(vm, "%s: %s handle has already been freed", who,
                      kind_name(kind));
  return handle;
}

LlmHandle* HandleTable::find(Scheme::VM* vm, Scheme::ValueT* value,
                             LlmKind kind)
{
  if (!value || !Scheme::scm_is_lightuserdata(vm, value))
    return NULL;

  void* raw = Scheme::scm_to_lightuserdata(vm, value);
  if (!issued(raw))
    return NULL;

  LlmHandle* handle = static_cast<LlmHandle*>(raw);
  if (handle->kind != kind || !handle->alive)
    return NULL;
  return handle;
}

void HandleTable::release(LlmHandle* handle)
{
  if (!handle || !handle->alive)
    return;
  if (handle->destroy)
    handle->destroy(handle->ptr);
  handle->ptr = NULL;
  handle->alive = false;
}

void HandleTable::shutdown()
{
  // Newest first: contexts and batches must go before the model they use.
  for (size_t i = handles_.size(); i > 0; --i)
    release(handles_[i - 1]);
  for (size_t i = 0; i < handles_.size(); ++i)
    delete handles_[i];
  handles_.clear();
}

} // namespace scmllama
