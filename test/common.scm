;; Shared helpers for the test scripts.
;;
;; Loaded with (load "test/common.scm"), so run from the repository root.

;; Silence llama.cpp diagnostics (model load info, metal pipelines, ...).
;; Comment out or call (llm-quiet! #f) when debugging the C side.
(llm-quiet! #t)

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
