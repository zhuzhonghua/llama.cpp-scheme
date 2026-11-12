// scmllama_sampler.cpp — llama_sampler bindings.
//
// Scheme side:
//   (llm-sampler-new model [opts])   -> sampler
//   (llm-sampler-sample smpl ctx)    -> token id
//   (llm-sampler-free smpl)          -> #t
//   (llm-sampler? smpl)              -> boolean
//
// Options for llm-sampler-new:
//   greedy          boolean  #f      greedy instead of dist
//   seed            integer  random  PRNG seed used by dist
//   temp            number   1.0     temperature (<= 0 keeps only the max)
//   top-k           integer  0       keep the k best logits (0 = off)
//   top-p           number   1.0     nucleus sampling (1.0 = off)
//   min-p           number   0.0     min-p sampling (0.0 = off)
//   penalty-last-n  integer  0       tokens to penalize (0 = off)
//   penalty-repeat  number   1.0
//   penalty-freq    number   0.0
//   penalty-present number   0.0
//   grammar         string  none     GBNF source; constrains sampled tokens
//   grammar-root    string  "root"   start symbol inside the grammar

#include "scmllama_util.h"
#include "scmllama_handle.h"

#include "llama.h"

namespace scmllama {

static void scmllama_sampler_destroy(void* ptr)
{
  if (ptr)
    llama_sampler_free(static_cast<llama_sampler*>(ptr));
}

// Builds the usual pipeline: penalties -> top-k -> top-p -> min-p -> temp ->
// greedy/dist.  The last stage selects the token.
static Scheme::ValueT scmllama_stub_llm_sampler_new(Scheme::VM* vm,
                                                    Scheme::ValueT* value,
                                                    Scheme::ValueT* rest)
{
  static const char* const kWho = "llm-sampler-new";
  static const char* const kKeys[] = {
    "greedy", "seed", "temp",
    "top-k", "top-p", "min-p",
    "penalty-last-n", "penalty-repeat", "penalty-freq", "penalty-present",
    "grammar", "grammar-root"
  };

  Scheme::ValueT* opts = Util::option_from_rest(vm, rest, kWho);
  const llama_vocab* vocab = Util::model_vocab(vm, value, kWho);
  int32_t n_vocab = llama_vocab_n_tokens(vocab);

  Util::option_require_known(
      vm, opts, kKeys,
      static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0])), kWho);

  bool greedy = Util::option_get_bool(vm, opts, "greedy", false, kWho);

  int64_t seed = Util::option_get_int(vm, opts, "seed", LLAMA_DEFAULT_SEED, kWho);
  if (seed < 0 || seed > static_cast<int64_t>(UINT32_MAX))
    Util::error_raise(vm, "%s: seed %lld is out of range", kWho,
                      static_cast<long long>(seed));

  float temp = Util::option_get_float(vm, opts, "temp", 1.0f, kWho);

  int64_t top_k = Util::option_get_int(vm, opts, "top-k", 0, kWho);
  if (top_k < 0 || top_k > INT32_MAX)
    Util::error_raise(vm, "%s: top-k %lld is out of range", kWho,
                      static_cast<long long>(top_k));

  float top_p = Util::option_get_float(vm, opts, "top-p", 1.0f, kWho);
  float min_p = Util::option_get_float(vm, opts, "min-p", 0.0f, kWho);
  if (top_p < 0.0f || top_p > 1.0f)
    Util::error_raise(vm, "%s: top-p must be within [0, 1]", kWho);
  if (min_p < 0.0f || min_p > 1.0f)
    Util::error_raise(vm, "%s: min-p must be within [0, 1]", kWho);

  int64_t penalty_last_n =
      Util::option_get_int(vm, opts, "penalty-last-n", 0, kWho);
  if (penalty_last_n < 0 || penalty_last_n > INT32_MAX)
    Util::error_raise(vm, "%s: penalty-last-n %lld is out of range", kWho,
                      static_cast<long long>(penalty_last_n));

  float penalty_repeat =
      Util::option_get_float(vm, opts, "penalty-repeat", 1.0f, kWho);
  float penalty_freq = Util::option_get_float(vm, opts, "penalty-freq", 0.0f, kWho);
  float penalty_present =
      Util::option_get_float(vm, opts, "penalty-present", 0.0f, kWho);

  // Grammar source and start symbol.  The sampler is created below, once the
  // chain exists, because it needs the model vocabulary.
  std::string grammar_storage;
  const char* grammar_str = NULL;
  Scheme::ValueT* grammar_slot = Util::option_find(vm, opts, "grammar", kWho);
  if (grammar_slot)
  {
    if (!Scheme::scm_is_string(vm, grammar_slot))
      Util::error_raise(vm, "%s: option 'grammar' expects a string, got %s",
                        kWho, Scheme::scm_typename(vm, grammar_slot));
    int len = 0;
    const char* chars = Scheme::scm_get_string(vm, grammar_slot, &len);
    grammar_storage.assign(chars ? chars : "", chars ? len : 0);
    if (grammar_storage.empty())
      Util::error_raise(vm, "%s: option 'grammar' is empty", kWho);
    grammar_str = grammar_storage.c_str();
  }

  std::string grammar_root_storage("root");
  Scheme::ValueT* grammar_root_slot =
      Util::option_find(vm, opts, "grammar-root", kWho);
  if (grammar_root_slot)
  {
    if (!Scheme::scm_is_string(vm, grammar_root_slot))
      Util::error_raise(vm, "%s: option 'grammar-root' expects a string, got %s",
                        kWho, Scheme::scm_typename(vm, grammar_root_slot));
    int len = 0;
    const char* chars = Scheme::scm_get_string(vm, grammar_root_slot, &len);
    grammar_root_storage.assign(chars ? chars : "", chars ? len : 0);
    if (grammar_root_storage.empty())
      Util::error_raise(vm, "%s: option 'grammar-root' is empty", kWho);
  }

  llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
  llama_sampler* chain = llama_sampler_chain_init(sparams);
  if (!chain)
    Util::error_raise(vm, "%s: failed to create sampler chain", kWho);

  // Mask disallowed tokens before any distribution filter runs.
  if (grammar_str)
  {
    llama_sampler* grammar = llama_sampler_init_grammar(
        vocab, grammar_str, grammar_root_storage.c_str());
    if (!grammar)
    {
      llama_sampler_free(chain);
      Util::error_raise(vm, "%s: failed to parse grammar", kWho);
    }
    llama_sampler_chain_add(chain, grammar);
  }

  if (penalty_last_n > 0)
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(
        n_vocab, static_cast<int32_t>(penalty_last_n), penalty_repeat,
        penalty_freq, penalty_present));

  if (top_k > 0)
    llama_sampler_chain_add(chain,
        llama_sampler_init_top_k(static_cast<int32_t>(top_k)));

  if (top_p < 1.0f)
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));

  if (min_p > 0.0f)
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));

  if (greedy)
  {
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
  }
  else
  {
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain,
        llama_sampler_init_dist(static_cast<uint32_t>(seed)));
  }

  return Util::handletable(vm)->make(vm, kLlmSampler, chain,
                                     scmllama_sampler_destroy);
}

// Samples from the logits of the last token decoded into ctx.
static Scheme::ValueT scmllama_stub_llm_sampler_sample(Scheme::VM* vm,
                                                       Scheme::ValueT* value,
                                                       Scheme::ValueT* ctx_value)
{
  static const char* const kWho = "llm-sampler-sample";

  LlmHandle* smpl_handle =
      Util::handletable(vm)->get(vm, value, kLlmSampler, kWho);
  llama_context* ctx = Util::context_ptr(vm, ctx_value, kWho);

  llama_token token = llama_sampler_sample(
      static_cast<llama_sampler*>(smpl_handle->ptr), ctx, -1);
  if (token < 0)
    Util::error_raise(vm, "%s: sampling failed", kWho);

  return Scheme::scm_make_int(vm, token);
}

static Scheme::ValueT scmllama_stub_llm_sampler_free(Scheme::VM* vm,
                                                     Scheme::ValueT* value)
{
  LlmHandle* handle = Util::handletable(vm)->peek(vm, value, kLlmSampler,
                                                  "llm-sampler-free");
  Util::handletable(vm)->release(handle);
  return Scheme::scm_make_bool(vm, true);
}

static Scheme::ValueT scmllama_stub_llm_sampler_p(Scheme::VM* vm,
                                                  Scheme::ValueT* value)
{
  return Scheme::scm_make_bool(
      vm, Util::handletable(vm)->find(vm, value, kLlmSampler) != NULL);
}

void scmllama_register_sampler_lib(Scheme::VM* vm)
{
  static const Scheme::RegCProc kLib[] = {
    Scheme::RegCProc("llm-sampler-new",    scmllama_stub_llm_sampler_new, true),
    Scheme::RegCProc("llm-sampler-sample", scmllama_stub_llm_sampler_sample),
    Scheme::RegCProc("llm-sampler-free",   scmllama_stub_llm_sampler_free),
    Scheme::RegCProc("llm-sampler?",       scmllama_stub_llm_sampler_p),
    Scheme::RegCProc(NULL, -1)
  };
  Scheme::scm_register_lib(vm, kLib);
}

} // namespace scmllama
