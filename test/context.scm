;; Context binding smoke test.
;;
;;   ./build/scmllama test/context.scm

(require "test.common")

;; ---- context lifecycle --------------------------------------------------

(define m (llm-model-load model-path))

(define ctx (llm-context-new m '((n-ctx . 512) (n-batch . 256))))

(display "context? ") (display ": ") (write (llm-context? ctx)) (newline)
(display "context? rejects non-handle") (display ": ") (write (llm-context? 42)) (newline)

(define info (llm-context-info ctx))
(display "n-ctx          : ") (display (hash-table-ref info "n-ctx")) (newline)
(display "n-batch        : ") (display (hash-table-ref info "n-batch")) (newline)
(display "n-seq-max      : ") (display (hash-table-ref info "n-seq-max")) (newline)
(display "n-threads      : ") (display (hash-table-ref info "n-threads")) (newline)
(display "n-threads-batch: ") (display (hash-table-ref info "n-threads-batch")) (newline)
(display "pooling-type   : ") (display (hash-table-ref info "pooling-type")) (newline)

;; Threads can be changed on a live context; second count is optional.
(display "set-threads 2/3: ") (display (llm-set-threads ctx 2 3)) (newline)
(define info2 (llm-context-info ctx))
(display "  -> n-threads/n-threads-batch: ")
(display (hash-table-ref info2 "n-threads")) (display "/")
(display (hash-table-ref info2 "n-threads-batch")) (newline)
(display "set-threads 1 (batch kept): ") (display (llm-set-threads ctx 1)) (newline)
(define info3 (llm-context-info ctx))
(display "  -> n-threads/n-threads-batch: ")
(display (hash-table-ref info3 "n-threads")) (display "/")
(display (hash-table-ref info3 "n-threads-batch")) (newline)

(display "n-ctx is 512") (display ": ") (write (hash-table-ref info "n-ctx")) (newline)
(display "n-batch is 256") (display ": ") (write (hash-table-ref info "n-batch")) (newline)
(display "n-seq-max is positive") (display ": ") (write (> (hash-table-ref info "n-seq-max") 0)) (newline)

;; ---- error paths --------------------------------------------------------

;; Unknown key must be rejected.
(define bad-key
  (xpcall (lambda () (llm-context-new m '((bad-key . 1))))
          (lambda (e) e)))
(display "unknown key raises error") (display ": ") (write bad-key) (newline)

;; Free is idempotent.
(display "free is idempotent") (display ": ") (write (llm-context-free ctx)) (newline)
(display "free is idempotent (second call)") (display ": ") (write (llm-context-free ctx)) (newline)
(display "freed handle is no longer a context") (display ": ") (write (llm-context? ctx)) (newline)

;; Use-after-free must be a readable Scheme error, not a crash.
(define uaf (xpcall (lambda () (llm-context-info ctx)) (lambda (e) e)))
(display "use-after-free raises an error") (display ": ") (write uaf) (newline)

;; Model without context is not a context.
(display "model is not a context") (display ": ") (write (llm-context? m)) (newline)

(llm-model-free m)

(display "done") (newline)
