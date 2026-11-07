# llama.cpp-scheme

Scheme bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), built on [dummyscheme](https://github.com/zhuzhonghua/dummyscheme) — a small, embeddable Scheme VM.

The project produces a host executable, `scmllama`, that loads GGUF models and runs local LLM inference directly from Scheme scripts: tokenization, decoding, sampling, and text generation, with full lifecycle control over handles (model / context / batch / sampler). Errors surface as readable Scheme errors instead of process crashes.

## Build & Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/scmllama test/hello.scm
```

## API

All functions are registered as native Scheme procedures (prefix `llm-`). Optional parameters are passed as an alist, e.g. `(llm-model-load "model.gguf" '(n-gpu-layers . 99))`.

### General

| Function | Description |
| --- | --- |
| `(llm-version)` | llama.cpp version string |
| `(llm-system-info)` | Build / system information |
| `(llm-quiet! bool)` | Silence llama.cpp and ggml logging |

### Model

| Function | Description |
| --- | --- |
| `(llm-model-load path [options...])` | Load a GGUF model from file |
| `(llm-model-free model)` | Release a model |
| `(llm-model? obj)` | Type predicate |
| `(llm-model-info model)` | Model metadata as a hash-table (size, layers, vocab, tokens, chat template, ...) |

### Vocab / Tokenization

| Function | Description |
| --- | --- |
| `(llm-tokenize model text [options...])` | Text → tokens |
| `(llm-detokenize model tokens [options...])` | Tokens → text |
| `(llm-token-to-piece model token)` | Token → raw piece string |
| `(llm-eog? model token)` | End-of-generation token check |

### Context

| Function | Description |
| --- | --- |
| `(llm-context-new model [options...])` | Create an inference context |
| `(llm-context-free ctx)` | Release a context |
| `(llm-context? obj)` | Type predicate |
| `(llm-context-info ctx)` | Context info |
| `(llm-reset ctx)` | Clear the KV cache / memory |

### Batch / Decode

| Function | Description |
| --- | --- |
| `(llm-batch-from-tokens tokens [options...])` | Build a batch from a token list |
| `(llm-batch-set-logits batch index)` | Mark a position to produce logits |
| `(llm-batch-n-tokens batch)` | Number of tokens in the batch |
| `(llm-batch-free batch)` | Release a batch |
| `(llm-batch? obj)` | Type predicate |
| `(llm-decode ctx batch)` | Run the forward pass |

### Sampling

| Function | Description |
| --- | --- |
| `(llm-sampler-new model [options...])` | Create a sampler chain (penalties → top-k → top-p → min-p → temp → greedy/dist) |
| `(llm-sampler-sample sampler ctx)` | Sample the next token from logits |
| `(llm-sampler-free sampler)` | Release a sampler |
| `(llm-sampler? obj)` | Type predicate |

### Generation

| Function | Description |
| --- | --- |
| `(llm-generate ctx sampler prompt callback [options...])` | Synchronous generation loop; returns generated text, calls `callback` per token piece, stops on EOG or `max-tokens` |

## License

See [LICENSE](LICENSE).
