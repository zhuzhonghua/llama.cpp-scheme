;; Shared helpers for the test scripts.
;;
;; Required with (require "test.common") — run from the repository root so
;; the default require path "./?.scm" resolves to ./test/common.scm.

;; llama.cpp and ggml diagnostics are disabled before the backend starts (see
;; Runtime::Runtime), so no llm-quiet! call is needed here.  Call
;; (llm-quiet! #f) if you want to see logs emitted after startup.

(define model-path "models/Qwen3-0.6B-Q4_K_M.gguf")

(define (check name got want)
  (if (equal? got want)
      (begin
        (display "PASS ")
        (display name)
        (newline))
      (begin
        (display "FAIL ")
        (display name)
        (display ": got ")
        (write got)
        (display ", want ")
        (write want)
        (newline)
        (error "assertion failed"))))

(define (check-true name value)
  (check name (if value #t #f) #t))
