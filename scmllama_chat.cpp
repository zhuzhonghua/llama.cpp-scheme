// Chat template bindings.
//
// Scheme side:
//   (llm-chat-template-apply model messages [opts]) -> string
//   (llm-chat-templates)                            -> vector of names
//
// `messages` is a proper list of (role . content) pairs, e.g.
//   '(("system" . "You are helpful.") ("user" . "Hello"))
//
// Options:
//   template      string  model's built-in template  template source or name
//   add-assistant boolean #t                         append the assistant turn
//
// The template is rendered by llama_chat_apply_template(), which recognises
// a fixed set of common templates (see (llm-chat-templates)).  Models that
// carry a chat template in their GGUF metadata use it by default; otherwise
// the caller must pass one explicitly.

#include "scmllama_util.h"
#include "scmllama_handle.h"

#include "llama.h"

#include <string>
#include <vector>

namespace scmllama {

// Names of the templates the built-in renderer understands.
static Scheme::ValueT scmllama_stub_llm_chat_templates(Scheme::VM* vm)
{
  const int32_t count = llama_chat_builtin_templates(NULL, 0);

  Sgcvar1(vm, vec);
  *vec = Scheme::scm_make_vector(vm, count > 0 ? static_cast<int>(count) : 0);
  if (count <= 0)
    return *vec;

  std::vector<const char*> names(static_cast<size_t>(count), NULL);
  llama_chat_builtin_templates(names.data(), names.size());

  for (int32_t i = 0; i < count; ++i)
  {
    Sgcvar1(vm, name);
    *name = Scheme::scm_make_string(vm, names[i] ? names[i] : "");
    Scheme::scm_vector_set(vm, vec, static_cast<int>(i), *name);
  }

  return *vec;
}

// Render a message list into a prompt string with the model's chat template.
static Scheme::ValueT scmllama_stub_llm_chat_template_apply(
    Scheme::VM* vm, Scheme::ValueT* model_value, Scheme::ValueT* messages_value,
    Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-chat-template-apply";
  static const char* const kKeys[] = {"template", "add-assistant"};

  // Validate the handle and options before touching the message list, so a
  // freed model reports the handle error rather than a message-shape error.
  const llama_model* model = Util::model_ptr(vm, model_value, kWho);

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);
  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  const bool add_assistant =
      Util::option_get_bool(vm, opts, "add-assistant", true, kWho);

  std::string template_storage;
  const char* chat_template = NULL;

  Scheme::ValueT* template_slot = Util::option_find(vm, opts, "template", kWho);
  if (template_slot)
  {
    if (!Scheme::scm_is_string(vm, template_slot))
      Util::error_raise(vm, "%s: option 'template' expects a string, got %s",
                        kWho, Scheme::scm_typename(vm, template_slot));
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
                        kWho);
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
                        kWho, index, Scheme::scm_typename(vm, message));

    Scheme::ValueT* role = Scheme::scm_car(vm, message);
    Scheme::ValueT* content = Scheme::scm_cdr(vm, message);

    if (!Scheme::scm_is_string(vm, role))
      Util::error_raise(vm, "%s: message %d role must be a string, got %s",
                        kWho, index, Scheme::scm_typename(vm, role));
    if (!Scheme::scm_is_string(vm, content))
      Util::error_raise(vm,
                        "%s: message %d must be a (role . content) pair "
                        "(content is not a string: %s)",
                        kWho, index, Scheme::scm_typename(vm, content));

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
    Util::error_raise(vm, "%s: messages must be a proper list", kWho);

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
                      kWho);

  if (written >= static_cast<int32_t>(buf.size()))
  {
    buf.resize(static_cast<size_t>(written) + 1);
    written = llama_chat_apply_template(
        chat_template, chat.data(), chat.size(), add_assistant, buf.data(),
        static_cast<int32_t>(buf.size()));
    if (written < 0 || written >= static_cast<int32_t>(buf.size()))
      Util::error_raise(vm, "%s: failed to render the chat template", kWho);
  }

  return Scheme::scm_make_string(vm, buf.data(), written);
}

void scmllama_register_chat_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-chat-template-apply",
                     scmllama_stub_llm_chat_template_apply, true),
    Scheme::RegCProc("llm-chat-templates", scmllama_stub_llm_chat_templates),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
