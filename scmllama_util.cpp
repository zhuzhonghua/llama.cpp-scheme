#include "scmllama_handle.h"
#include "scmllama_util.h"

#include "llama.h"

#include <cstdarg>

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

// ---- Util: model ----

const llama_model* Util::model_ptr(LlmHandle* handle)
{
  return static_cast<const llama_model*>(handle->ptr);
}

const llama_vocab* Util::model_vocab(LlmHandle* handle)
{
  return llama_model_get_vocab(model_ptr(handle));
}

} // namespace scmllama
