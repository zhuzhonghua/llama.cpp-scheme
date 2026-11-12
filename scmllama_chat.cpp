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

  const std::string prompt =
      Util::chat_render(vm, model, messages_value, opts, kWho);
  return Scheme::scm_make_string(vm, prompt.data(),
                                 static_cast<int>(prompt.size()));
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
