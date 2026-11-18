;; Batch binding smoke test.
;;
;;   ./build/scmllama test/batch.scm

(require "test.common")

;; ---- batch lifecycle ----------------------------------------------------

(define m (llm-model-load model-path))
(define ctx (llm-context-new m '((n-ctx . 512))))

;; batch from token list
(define tokens (llm-tokenize m "Once upon a time" '((add-special . #f))))
(define batch (llm-batch-from-tokens tokens '((last-logits . #t))))

(display "batch? ") (display ": ") (write (llm-batch? batch)) (newline)
(display "batch n-tokens") (display ": ") (write (llm-batch-n-tokens batch)) (newline)
(display "batch? rejects non-batch") (display ": ") (write (llm-batch? 42)) (newline)

;; set-logits on first token
(display "set-logits") (display ": ") (write (llm-batch-set-logits batch 0 #t)) (newline)

;; decode should succeed
(display "decode") (display ": ") (write (llm-decode ctx batch)) (newline)
(llm-context-free ctx)

;; batch from token list, logits off; decode on a fresh context
;; (KV cache requires consecutive positions per sequence)
(define batch2 (llm-batch-from-tokens (vector->list tokens) '((last-logits . #f))))
(display "batch2 n-tokens") (display ": ") (write (llm-batch-n-tokens batch2)) (newline)
(define ctx2 (llm-context-new m '((n-ctx . 512))))
(display "decode batch2") (display ": ") (write (llm-decode ctx2 batch2)) (newline)

;; ---- error paths --------------------------------------------------------

;; empty list must fail
(define empty-batch
  (xpcall (lambda () (llm-batch-from-tokens '()))
          (lambda (e) e)))
(display "empty token list raises error") (display ": ") (write empty-batch) (newline)

;; unknown key in opts must fail
(define bad-key
  (xpcall (lambda () (llm-batch-from-tokens tokens '((bad-key . 1))))
          (lambda (e) e)))
(display "unknown key raises error") (display ": ") (write bad-key) (newline)

;; free is idempotent
(display "free is idempotent") (display ": ") (write (llm-batch-free batch)) (newline)
(display "free is idempotent (second call)") (display ": ") (write (llm-batch-free batch)) (newline)
(display "freed batch is no longer a batch") (display ": ") (write (llm-batch? batch)) (newline)

;; use-after-free must raise error
(define uaf (xpcall (lambda () (llm-batch-n-tokens batch)) (lambda (e) e)))
(display "use-after-free raises error") (display ": ") (write uaf) (newline)

;; batch from tokens with non-integer must fail
(define bad-tokens
  (xpcall (lambda () (llm-batch-from-tokens '(1 "two" 3)))
          (lambda (e) e)))
(display "non-integer token raises error") (display ": ") (write bad-tokens) (newline)

;; batch from tokens with out-of-range id must fail
(define oor-tokens
  (xpcall (lambda () (llm-batch-from-tokens '(1 -5 3)))
          (lambda (e) e)))
(display "negative token id raises error") (display ": ") (write oor-tokens) (newline)

(llm-batch-free batch2)
(llm-context-free ctx2)
(llm-model-free m)

(display "done") (newline)
