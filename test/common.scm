;; Shared helpers for the test scripts.
;;
;; Required with (require "test.common") — run from the repository root so
;; the default require path "./?.scm" resolves to ./test/common.scm.

;; llama.cpp and ggml diagnostics are disabled before the backend starts (see
;; Runtime::Runtime), so no llm-quiet! call is needed here.  Call
;; (llm-quiet! #f) if you want to see logs emitted after startup.

(define model-path "models/Qwen3-0.6B-Q4_K_M.gguf")

;; Scheme test scripts are code examples only (2026-11-18): they print values
;; with display/write for visual inspection and never assert PASS/FAIL.  The
;; old `check` / `check-true` helpers were removed when all scripts were
;; converted — do not reintroduce them.  See context.md section 8.
