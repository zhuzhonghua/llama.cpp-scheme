// Vocabulary (tokenizer) bindings.
//
// Scheme side:
//   (llm-tokenize model text [opts])         -> vector of token ids
//   (llm-detokenize model tokens [opts])     -> string
//   (llm-token-to-piece model token [opts])  -> string
//   (llm-eog? model token)                   -> boolean

#include "scmllama.h"
#include "scmllama_util.h"

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scmllama {

// --- token helpers ---

static llama_token scmllama_token_from_value(Scheme::VM* vm,
                                             Scheme::ValueT* value,
                                             const char* who)
{
  if (!value || !Scheme::scm_is_int(vm, value))
  {
    Util::error_raise(vm, "%s: expected a token id (integer), got %s", who,
                 value ? Scheme::scm_typename(vm, value) : "nothing");
  }

  int64_t id = static_cast<int64_t>(Scheme::scm_get_int(vm, value));
  if (id < 0 || id > INT32_MAX)
    Util::error_raise(vm, "%s: token id %lld is out of range", who,
                 static_cast<long long>(id));
  return static_cast<llama_token>(id);
}

static std::vector<llama_token> scmllama_token_list(Scheme::VM* vm,
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
      tokens.push_back(scmllama_token_from_value(vm, &token, who));
    }
    return tokens;
  }

  if (Scheme::scm_is_list(vm, value))
  {
    Scheme::ValueT* cur = value;
    while (Scheme::scm_is_pair(vm, cur))
    {
      tokens.push_back(
          scmllama_token_from_value(vm, Scheme::scm_car(vm, cur), who));
      cur = Scheme::scm_cdr(vm, cur);
    }
    if (!Scheme::scm_is_null(vm, cur))
      Util::error_raise(vm, "%s: expected a proper list of token ids", who);
    return tokens;
  }

  Util::error_raise(vm, "%s: expected a vector or list of token ids, got %s", who,
               value ? Scheme::scm_typename(vm, value) : "nothing");
}

// --- stubs ---

static Scheme::ValueT scmllama_stub_llm_tokenize(Scheme::VM* vm,
                                                 Scheme::ValueT* value,
                                                 Scheme::ValueT* text_value,
                                                 Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-tokenize";
  static const char* const kKeys[] = {"add-special", "parse-special"};

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);
  LlmHandle* handle = Util::model_handle(vm, value, kWho);
  const llama_vocab* vocab = Util::model_vocab(handle);

  int text_len = 0;
  const char* text = Scheme::scm_get_string(vm, text_value, &text_len);

  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  bool add_special = Util::option_get_bool(vm, opts, "add-special", true, kWho);
  bool parse_special =
      Util::option_get_bool(vm, opts, "parse-special", false, kWho);

  // Upper bound: one token per byte, plus up to two special tokens.
  int32_t cap = text_len + (add_special ? 2 : 0);
  std::vector<llama_token> tokens(static_cast<size_t>(cap));
  int32_t count = llama_tokenize(vocab, text, text_len, tokens.data(), cap,
                                 add_special, parse_special);
  if (count == INT32_MIN)
    Util::error_raise(vm, "%s: tokenization result exceeds int32 limit", kWho);
  if (count < 0)
  {
    cap = -count;
    tokens.resize(static_cast<size_t>(cap));
    count = llama_tokenize(vocab, text, text_len, tokens.data(), cap,
                           add_special, parse_special);
    if (count < 0)
      Util::error_raise(vm, "%s: tokenization failed", kWho);
  }
  tokens.resize(static_cast<size_t>(count));

  Sgcvar1(vm, result);
  *result = Scheme::scm_make_vector(vm, static_cast<int>(tokens.size()));
  for (size_t i = 0; i < tokens.size(); ++i)
  {
    Scheme::scm_vector_set(vm, result, static_cast<int>(i),
                           Scheme::scm_make_int(vm, tokens[i]));
  }
  return *result;
}

static Scheme::ValueT scmllama_stub_llm_detokenize(Scheme::VM* vm,
                                                   Scheme::ValueT* value,
                                                   Scheme::ValueT* tokens_value,
                                                   Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-detokenize";
  static const char* const kKeys[] = {"remove-special", "unparse-special"};

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);
  LlmHandle* handle = Util::model_handle(vm, value, kWho);
  const llama_vocab* vocab = Util::model_vocab(handle);

  std::vector<llama_token> tokens =
      scmllama_token_list(vm, tokens_value, kWho);

  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  bool remove_special =
      Util::option_get_bool(vm, opts, "remove-special", true, kWho);
  bool unparse_special =
      Util::option_get_bool(vm, opts, "unparse-special", false, kWho);

  if (tokens.empty())
    return Scheme::scm_make_string(vm, "", 0);

  // A piece is usually a few bytes; start with a guess and grow on demand.
  int32_t cap = static_cast<int32_t>(tokens.size()) * 4;
  if (cap < 256)
    cap = 256;

  std::vector<char> text(static_cast<size_t>(cap));
  int32_t count =
      llama_detokenize(vocab, tokens.data(), static_cast<int32_t>(tokens.size()),
                       text.data(), cap, remove_special, unparse_special);
  if (count == INT32_MIN)
    Util::error_raise(vm, "%s: detokenization result exceeds int32 limit", kWho);
  if (count < 0)
  {
    cap = -count;
    text.resize(static_cast<size_t>(cap));
    count = llama_detokenize(vocab, tokens.data(),
                             static_cast<int32_t>(tokens.size()), text.data(),
                             cap, remove_special, unparse_special);
    if (count < 0)
      Util::error_raise(vm, "%s: detokenization failed", kWho);
  }

  return Scheme::scm_make_string(vm, text.data(), count);
}

static Scheme::ValueT scmllama_stub_llm_token_to_piece(
    Scheme::VM* vm, Scheme::ValueT* value, Scheme::ValueT* token_value,
    Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-token-to-piece";
  static const char* const kKeys[] = {"lstrip", "special"};

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);
  LlmHandle* handle = Util::model_handle(vm, value, kWho);
  const llama_vocab* vocab = Util::model_vocab(handle);
  llama_token token = scmllama_token_from_value(vm, token_value, kWho);

  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  int64_t lstrip = Util::option_get_int(vm, opts, "lstrip", 0, kWho);
  if (lstrip < 0 || lstrip > INT32_MAX)
    Util::error_raise(vm, "%s: lstrip %lld is out of range", kWho,
                 static_cast<long long>(lstrip));
  bool special = Util::option_get_bool(vm, opts, "special", false, kWho);

  std::vector<char> buf(256);
  int32_t count = llama_token_to_piece(vocab, token, buf.data(), buf.size(),
                                       static_cast<int32_t>(lstrip), special);
  if (count < 0)
  {
    buf.resize(static_cast<size_t>(-count));
    count = llama_token_to_piece(vocab, token, buf.data(), buf.size(),
                                 static_cast<int32_t>(lstrip), special);
    if (count < 0)
      Util::error_raise(vm, "%s: could not convert token %d to a piece", kWho,
                   static_cast<int>(token));
  }

  return Scheme::scm_make_string(vm, buf.data(), count);
}

static Scheme::ValueT scmllama_stub_llm_eog_p(Scheme::VM* vm,
                                              Scheme::ValueT* value,
                                              Scheme::ValueT* token_value)
{
  LlmHandle* handle = Util::model_handle(vm, value, "llm-eog?");
  const llama_vocab* vocab = Util::model_vocab(handle);
  llama_token token = scmllama_token_from_value(vm, token_value, "llm-eog?");
  return Scheme::scm_make_bool(vm, llama_vocab_is_eog(vocab, token));
}

void scmllama_register_vocab_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-tokenize",       scmllama_stub_llm_tokenize, true),
    Scheme::RegCProc("llm-detokenize",     scmllama_stub_llm_detokenize, true),
    Scheme::RegCProc("llm-token-to-piece", scmllama_stub_llm_token_to_piece,
                     true),
    Scheme::RegCProc("llm-eog?",           scmllama_stub_llm_eog_p),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
