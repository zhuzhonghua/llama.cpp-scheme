;; Context binding smoke test.
;;
;;   ./build/scmllama test/context.scm
;;
;; Uses Qwen3-0.6B (models/Qwen3-0.6B-Q4_K_M.gguf).

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

;; ---- context lifecycle --------------------------------------------------

(define m (llm-model-load model-path))

(define ctx (llm-context-new m '((n-ctx . 512) (n-batch . 256))))

(check-true "context? " (llm-context? ctx))
(check "context? rejects non-handle" (llm-context? 42) #f)

(define info (llm-context-info ctx))
(display "n-ctx    : ") (display (hash-table-ref info "n-ctx")) (newline)
(display "n-batch  : ") (display (hash-table-ref info "n-batch")) (newline)
(display "n-seq-max: ") (display (hash-table-ref info "n-seq-max")) (newline)

(check "n-ctx is 512" (hash-table-ref info "n-ctx") 512)
(check "n-batch is 256" (hash-table-ref info "n-batch") 256)
(check-true "n-seq-max is positive" (> (hash-table-ref info "n-seq-max") 0))

;; ---- error paths --------------------------------------------------------

;; Unknown key must be rejected.
(define bad-key
  (xpcall (lambda () (llm-context-new m '((bad-key . 1))))
          (lambda (e) e)))
(check-true "unknown key raises error" (string? bad-key))

;; Free is idempotent.
(check "free is idempotent" (llm-context-free ctx) #t)
(check "free is idempotent (second call)" (llm-context-free ctx) #t)
(check "freed handle is no longer a context" (llm-context? ctx) #f)

;; Use-after-free must be a readable Scheme error, not a crash.
(define uaf (xpcall (lambda () (llm-context-info ctx)) (lambda (e) e)))
(check-true "use-after-free raises an error" (string? uaf))

;; Model without context is not a context.
(check "model is not a context" (llm-context? m) #f)

(llm-model-free m)

(display "all context tests passed") (newline)
