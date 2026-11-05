;; Batch binding smoke test.
;;
;;   ./build/scmllama test/batch.scm
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

;; ---- batch lifecycle ----------------------------------------------------

(define m (llm-model-load model-path))
(define ctx (llm-context-new m '((n-ctx . 512))))

;; batch from token list
(define tokens (llm-tokenize m "Once upon a time" '((add-special . #f))))
(define batch (llm-batch-from-tokens tokens '((last-logits . #t))))

(check-true "batch? " (llm-batch? batch))
(check "batch n-tokens" (llm-batch-n-tokens batch) (vector-length tokens))
(check "batch? rejects non-batch" (llm-batch? 42) #f)

;; set-logits on first token
(check "set-logits" (llm-batch-set-logits batch 0 #t) #t)

;; decode should succeed
(check-true "decode" (llm-decode ctx batch))
(llm-context-free ctx)

;; batch from token list, logits off; decode on a fresh context
;; (KV cache requires consecutive positions per sequence)
(define batch2 (llm-batch-from-tokens (vector->list tokens) '((last-logits . #f))))
(check "batch2 n-tokens" (llm-batch-n-tokens batch2) (vector-length tokens))
(define ctx2 (llm-context-new m '((n-ctx . 512))))
(check-true "decode batch2" (llm-decode ctx2 batch2))

;; ---- error paths --------------------------------------------------------

;; empty list must fail
(define empty-batch
  (xpcall (lambda () (llm-batch-from-tokens '()))
          (lambda (e) e)))
(check-true "empty token list raises error" (string? empty-batch))

;; unknown key in opts must fail
(define bad-key
  (xpcall (lambda () (llm-batch-from-tokens tokens '((bad-key . 1))))
          (lambda (e) e)))
(check-true "unknown key raises error" (string? bad-key))

;; free is idempotent
(check "free is idempotent" (llm-batch-free batch) #t)
(check "free is idempotent (second call)" (llm-batch-free batch) #t)
(check "freed batch is no longer a batch" (llm-batch? batch) #f)

;; use-after-free must raise error
(define uaf (xpcall (lambda () (llm-batch-n-tokens batch)) (lambda (e) e)))
(check-true "use-after-free raises error" (string? uaf))

;; batch from tokens with non-integer must fail
(define bad-tokens
  (xpcall (lambda () (llm-batch-from-tokens '(1 "two" 3)))
          (lambda (e) e)))
(check-true "non-integer token raises error" (string? bad-tokens))

;; batch from tokens with out-of-range id must fail
(define oor-tokens
  (xpcall (lambda () (llm-batch-from-tokens '(1 -5 3)))
          (lambda (e) e)))
(check-true "negative token id raises error" (string? oor-tokens))

(llm-batch-free batch2)
(llm-context-free ctx2)
(llm-model-free m)

(display "all batch tests passed") (newline)
