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

LlmHandle* Util::context_handle(Scheme::VM* vm, Scheme::ValueT* value,
                                const char* who)
{
  return handletable(vm)->get(vm, value, kLlmContext, who);
}

LlmHandle* Util::sampler_handle(Scheme::VM* vm, Scheme::ValueT* value,
                                const char* who)
{
  return handletable(vm)->get(vm, value, kLlmSampler, who);
}

LlmHandle* Util::batch_handle(Scheme::VM* vm, Scheme::ValueT* value,
                              const char* who)
{
  return handletable(vm)->get(vm, value, kLlmBatch, who);
}

// ---- Util: model accessors ----

const llama_model* Util::model_ptr(Scheme::VM* vm, Scheme::ValueT* value,
                                   const char* who)
{
  return model_ptr(model_handle(vm, value, who));
}

const llama_vocab* Util::model_vocab(Scheme::VM* vm, Scheme::ValueT* value,
                                     const char* who)
{
  return model_vocab(model_handle(vm, value, who));
}

const llama_model* Util::model_ptr(LlmHandle* handle)
{
  return static_cast<const llama_model*>(handle->ptr);
}

const llama_vocab* Util::model_vocab(LlmHandle* handle)
{
  return llama_model_get_vocab(model_ptr(handle));
}

// ---- Util: context accessors ----

llama_context* Util::context_ptr(Scheme::VM* vm, Scheme::ValueT* value,
                                 const char* who)
{
  return context_ptr(context_handle(vm, value, who));
}

llama_context* Util::context_ptr(LlmHandle* handle)
{
  return static_cast<llama_context*>(handle->ptr);
}

// ---- Util: sampler accessors ----

llama_sampler* Util::sampler_ptr(Scheme::VM* vm, Scheme::ValueT* value,
                                 const char* who)
{
  return sampler_ptr(sampler_handle(vm, value, who));
}

llama_sampler* Util::sampler_ptr(LlmHandle* handle)
{
  return static_cast<llama_sampler*>(handle->ptr);
}

// ---- Util: batch accessors ----

llama_batch* Util::batch_ptr(Scheme::VM* vm, Scheme::ValueT* value,
                             const char* who)
{
  return batch_ptr(batch_handle(vm, value, who));
}

llama_batch* Util::batch_ptr(LlmHandle* handle)
{
  return static_cast<llama_batch*>(handle->ptr);
}

// ---- Util: KV cache ----

int64_t Util::kv_seq_used(llama_context* ctx, llama_seq_id seq_id)
{
  const llama_pos pos_max =
      llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
  return pos_max < 0 ? 0 : static_cast<int64_t>(pos_max) + 1;
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

std::vector<llama_token> Util::tokenize_text(Scheme::VM* vm,
                                             const llama_vocab* vocab,
                                             const char* text, int text_len,
                                             bool add_special,
                                             bool parse_special,
                                             const char* who)
{
  // Upper bound: one token per byte, plus up to two special tokens.
  int32_t cap = text_len + (add_special ? 2 : 0);
  std::vector<llama_token> tokens(static_cast<size_t>(cap));
  int32_t count = llama_tokenize(vocab, text, text_len, tokens.data(), cap,
                                 add_special, parse_special);
  if (count == INT32_MIN)
    error_raise(vm, "%s: tokenization result exceeds int32 limit", who);
  if (count < 0)
  {
    cap = -count;
    tokens.resize(static_cast<size_t>(cap));
    count = llama_tokenize(vocab, text, text_len, tokens.data(), cap,
                           add_special, parse_special);
    if (count < 0)
      error_raise(vm, "%s: tokenization failed", who);
  }
  tokens.resize(static_cast<size_t>(count));
  return tokens;
}

// ---- Util: chat ----

// Renders a message list into a prompt string.  The caller validates the
// option keys; this helper only reads `template` and `add-assistant`.
std::string Util::chat_render(Scheme::VM* vm, const llama_model* model,
                              Scheme::ValueT* messages_value,
                              Scheme::ValueT* opts, const char* who)
{
  const bool add_assistant =
      Util::option_get_bool(vm, opts, "add-assistant", true, who);

  std::string template_storage;
  const char* chat_template = NULL;

  Scheme::ValueT* template_slot = Util::option_find(vm, opts, "template", who);
  if (template_slot)
  {
    if (!Scheme::scm_is_string(vm, template_slot))
      Util::error_raise(vm, "%s: option 'template' expects a string, got %s",
                        who, Scheme::scm_typename(vm, template_slot));
    int len = 0;
    const char* chars = Scheme::scm_get_string(vm, template_slot, &len);
    template_storage.assign(chars ? chars : "", chars ? len : 0);
    chat_template = template_storage.c_str();
  }
  else
  {
    chat_template = llama_model_chat_template(model, NULL);
    if (!chat_template)
      Util::error_raise(vm,
                        "%s: model has no built-in chat template; pass one "
                        "with (template . \"chatml\")",
                        who);
  }

  // Walk the message list once.  Pointers returned by scm_car/scm_cdr stay
  // valid because messages_value is a rooted argument for the whole call.
  std::vector<std::string> roles;
  std::vector<std::string> contents;
  size_t total_chars = 0;

  int index = 0;
  Scheme::ValueT* cur = messages_value;
  while (Scheme::scm_is_pair(vm, cur))
  {
    Scheme::ValueT* message = Scheme::scm_car(vm, cur);
    if (!Scheme::scm_is_pair(vm, message))
      Util::error_raise(vm,
                        "%s: message %d must be a (role . content) pair, got %s",
                        who, index, Scheme::scm_typename(vm, message));

    Scheme::ValueT* role = Scheme::scm_car(vm, message);
    Scheme::ValueT* content = Scheme::scm_cdr(vm, message);

    if (!Scheme::scm_is_string(vm, role))
      Util::error_raise(vm, "%s: message %d role must be a string, got %s",
                        who, index, Scheme::scm_typename(vm, role));
    if (!Scheme::scm_is_string(vm, content))
      Util::error_raise(vm,
                        "%s: message %d must be a (role . content) pair "
                        "(content is not a string: %s)",
                        who, index, Scheme::scm_typename(vm, content));

    int role_len = 0;
    const char* role_chars = Scheme::scm_get_string(vm, role, &role_len);
    roles.push_back(std::string(role_chars ? role_chars : "",
                                role_chars ? role_len : 0));

    int content_len = 0;
    const char* content_chars = Scheme::scm_get_string(vm, content, &content_len);
    contents.push_back(std::string(content_chars ? content_chars : "",
                                   content_chars ? content_len : 0));

    total_chars += roles.back().size() + contents.back().size();

    cur = Scheme::scm_cdr(vm, cur);
    ++index;
  }
  if (!Scheme::scm_is_null(vm, cur))
    Util::error_raise(vm, "%s: messages must be a proper list", who);

  const size_t n_msg = roles.size();
  std::vector<llama_chat_message> chat(n_msg);
  for (size_t i = 0; i < n_msg; ++i)
  {
    chat[i].role = roles[i].c_str();
    chat[i].content = contents[i].c_str();
  }

  // One render pass.  llama_chat_apply_template() always formats the whole
  // string internally, so measuring first (buf = NULL) would do the work
  // twice.  Size the buffer from the message text, as the API recommends,
  // and retry only if the estimate turns out to be too small.
  std::vector<char> buf(2 * total_chars + 256, '\0');

  int32_t written = llama_chat_apply_template(
      chat_template, chat.data(), chat.size(), add_assistant, buf.data(),
      static_cast<int32_t>(buf.size()));

  if (written < 0)
    Util::error_raise(vm,
                      "%s: template is not supported by llama.cpp "
                      "(see llm-chat-templates)",
                      who);

  if (written >= static_cast<int32_t>(buf.size()))
  {
    buf.resize(static_cast<size_t>(written) + 1);
    written = llama_chat_apply_template(
        chat_template, chat.data(), chat.size(), add_assistant, buf.data(),
        static_cast<int32_t>(buf.size()));
    if (written < 0 || written >= static_cast<int32_t>(buf.size()))
      Util::error_raise(vm, "%s: failed to render the chat template", who);
  }

  return std::string(buf.data(), static_cast<size_t>(written));
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
