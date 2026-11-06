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

// ---- Util: token ----

llama_token Util::token_from_value(Scheme::VM* vm, Scheme::ValueT* value,
                                   const char* who)
{
  if (!value || !Scheme::scm_is_int(vm, value))
  {
    error_raise(vm, "%s: expected a token id (integer), got %s", who,
                value ? Scheme::scm_typename(vm, value) : "nothing");
  }

  int64_t id = static_cast<int64_t>(Scheme::scm_get_int(vm, value));
  if (id < 0 || id > INT32_MAX)
    error_raise(vm, "%s: token id %lld is out of range", who,
                static_cast<long long>(id));
  return static_cast<llama_token>(id);
}

std::vector<llama_token> Util::token_list(Scheme::VM* vm,
                                          Scheme::ValueT* value,
                                          const char* who)
{
  std::vector<llama_token> tokens;

  if (Scheme::scm_is_vector(vm, value))
  {
    int n = Scheme::scm_vector_len(vm, value);
    tokens.reserve(n);
    for (int i = 0; i < n; ++i)
    {
      Scheme::ValueT token = Scheme::scm_vector_ref(vm, value, i);
      tokens.push_back(token_from_value(vm, &token, who));
    }
    return tokens;
  }

  if (Scheme::scm_is_list(vm, value))
  {
    Scheme::ValueT* cur = value;
    while (Scheme::scm_is_pair(vm, cur))
    {
      tokens.push_back(token_from_value(vm, Scheme::scm_car(vm, cur), who));
      cur = Scheme::scm_cdr(vm, cur);
    }
    if (!Scheme::scm_is_null(vm, cur))
      error_raise(vm, "%s: expected a proper list of token ids", who);
    return tokens;
  }

  error_raise(vm, "%s: expected a vector or list of token ids, got %s", who,
              value ? Scheme::scm_typename(vm, value) : "nothing");
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

float Util::option_float(Scheme::VM* vm, Scheme::ValueT* slot, float def,
                         const char* who, const char* key)
{
  if (!slot || Scheme::scm_is_null(vm, slot))
    return def;
  if (Scheme::scm_is_int(vm, slot))
    return static_cast<float>(Scheme::scm_get_int(vm, slot));
  if (Scheme::scm_is_float(vm, slot))
    return static_cast<float>(Scheme::scm_get_float(vm, slot));
  error_raise(vm, "%s: option '%s' expects a number, got %s", who, key,
              Scheme::scm_typename(vm, slot));
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

float Util::option_get_float(Scheme::VM* vm, Scheme::ValueT* alist,
                             const char* key, float def, const char* who)
{
  return option_float(vm, option_find(vm, alist, key, who), def, who, key);
}

} // namespace scmllama
