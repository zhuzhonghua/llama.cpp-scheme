#include "scmllama_handle.h"
#include "scmllama_util.h"
#include "scmllama.h"

#include "llama.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace scmllama {

// ---- Util: error ----

void Util::error_raise(Scheme::VM* vm, const char* fmt, ...)
{
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  Scheme::scm_error(vm, "%s", buf);
  throw "ScmError";
}

// ---- Util: runtime ----

Runtime* Util::runtime(Scheme::VM* vm)
{
  Scheme::ValueT slot = Scheme::scm_registry_get(vm, Runtime::kRegistryKey);
  return static_cast<Runtime*>(Scheme::scm_to_lightuserdata(vm, &slot));
}

HandleTable* Util::handletable(Scheme::VM* vm)
{
  return runtime(vm)->handles();
}

// ---- Util: handle shortcuts ----

LlmHandle* Util::model_handle(Scheme::VM* vm, Scheme::ValueT* value,
                              const char* who)
{
  return handletable(vm)->get(vm, value, kLlmModel, who);
}

// ---- Util: model accessors ----

const llama_model* Util::model_ptr(LlmHandle* handle)
{
  return static_cast<const llama_model*>(handle->ptr);
}

const llama_vocab* Util::model_vocab(LlmHandle* handle)
{
  return llama_model_get_vocab(model_ptr(handle));
}

// ---- Util: option ----

Scheme::ValueT* Util::option_from_rest(Scheme::VM* vm, Scheme::ValueT* rest,
                                       const char* who)
{
  if (!rest || Scheme::scm_is_null(vm, rest))
    return NULL;

  int n = Scheme::scm_proper_list_len(vm, rest);
  if (n < 0 || n > 1)
    error_raise(vm, "%s: expected at most one options alist", who);
  return Scheme::scm_car(vm, rest);
}

Scheme::ValueT* Util::option_find(Scheme::VM* vm, Scheme::ValueT* alist,
                                  const char* key, const char* who)
{
  if (!alist || Scheme::scm_is_null(vm, alist))
    return NULL;

  Scheme::ValueT sym = Scheme::scm_make_symbol(vm, key);
  Scheme::ValueT* entry = Scheme::scm_assv(vm, &sym, alist);
  if (!entry || Scheme::scm_is_false(vm, entry))
    return NULL;
  return Scheme::scm_cdr(vm, entry);
}

void Util::option_require_known(Scheme::VM* vm, Scheme::ValueT* alist,
                                const char* const* keys, int count,
                                const char* who)
{
  if (!alist || Scheme::scm_is_null(vm, alist))
    return;

  Scheme::ValueT* cur = alist;
  while (Scheme::scm_is_pair(vm, cur))
  {
    Scheme::ValueT* entry = Scheme::scm_car(vm, cur);
    Scheme::ValueT* entry_key = Scheme::scm_car(vm, entry);

    int len = 0;
    const char* name = Scheme::scm_tostring(vm, entry_key, &len);
    bool known = false;
    for (int i = 0; i < count && name; ++i)
    {
      if (strcmp(name, keys[i]) == 0)
      {
        known = true;
        break;
      }
    }
    if (!known)
      error_raise(vm, "%s: unknown option '%s'", who, name ? name : "?");

    cur = Scheme::scm_cdr(vm, cur);
  }
}

bool Util::option_bool(Scheme::VM* vm, Scheme::ValueT* slot, bool def,
                      const char* who, const char* key)
{
  if (!slot || Scheme::scm_is_null(vm, slot))
    return def;
  if (!Scheme::scm_is_bool(vm, slot))
    error_raise(vm, "%s: option '%s' expects a boolean, got %s", who, key,
                Scheme::scm_typename(vm, slot));
  return Scheme::scm_get_bool(vm, slot);
}

int64_t Util::option_int(Scheme::VM* vm, Scheme::ValueT* slot, int64_t def,
                         const char* who, const char* key)
{
  if (!slot || Scheme::scm_is_null(vm, slot))
    return def;
  if (!Scheme::scm_is_int(vm, slot))
    error_raise(vm, "%s: option '%s' expects an integer, got %s", who, key,
                Scheme::scm_typename(vm, slot));
  return static_cast<int64_t>(Scheme::scm_get_int(vm, slot));
}

bool Util::option_get_bool(Scheme::VM* vm, Scheme::ValueT* alist,
                           const char* key, bool def, const char* who)
{
  return option_bool(vm, option_find(vm, alist, key, who), def, who, key);
}

int64_t Util::option_get_int(Scheme::VM* vm, Scheme::ValueT* alist,
                             const char* key, int64_t def, const char* who)
{
  return option_int(vm, option_find(vm, alist, key, who), def, who, key);
}

} // namespace scmllama
