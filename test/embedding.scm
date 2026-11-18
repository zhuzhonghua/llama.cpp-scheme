;; Embeddings binding test.
;;
;;   ./build/scmllama test/embedding.scm

(require "test.common")

(define m (llm-model-load model-path))
(define n-embd (hash-table-ref (llm-model-info m) "n-embd"))

;; ---- pooling none: last token embedding (default) ------------------------

(define ctx (llm-context-new m '((n-ctx . 512) (embeddings . #t))))

(define a (llm-embed ctx "Once upon a time"))
(display "dim (none) : ") (display (vector-length a)) (newline)

(display "embedding length is n-embd") (display ": ") (write (vector-length a)) (newline)
(display "embedding components are real numbers") (display ": ") (write (real? (vector-ref a 0))) (newline)

(define a2 (llm-embed ctx "Once upon a time"))
(display "same text yields the same vector") (display ": ") (write (equal? a a2)) (newline)

(define b (llm-embed ctx "The quick brown fox"))
(display "different text yields a different vector") (display ": ") (write (equal? a b)) (newline)

;; ---- mean pooling: one vector per sequence -------------------------------

(llm-context-free ctx)

(define pctx
  (llm-context-new m '((n-ctx . 512)
                       (embeddings . #t)
                       (pooling-type . mean))))

(define pooled (llm-embed pctx "Once upon a time"))
(display "dim (mean) : ") (display (vector-length pooled)) (newline)

(display "mean-pooled embedding length is n-embd") (display ": ") (write (vector-length pooled)) (newline)
(display "mean pooling differs from the last token") (display ": ") (write (not (equal? pooled a))) (newline)

;; ---- error paths ---------------------------------------------------------

(define plain (llm-context-new m '((n-ctx . 256))))

(define no-embeddings
  (xpcall (lambda () (llm-embed plain "hello")) (lambda (e) e)))
(display "context without embeddings raises an error") (display ": ") (write no-embeddings) (newline)

;; Empty text has no tokens to decode.
(define empty-text
  (xpcall (lambda () (llm-embed pctx "" '((add-special . #f))))
          (lambda (e) e)))
(display "empty text raises an error") (display ": ") (write empty-text) (newline)

(define bad-pooling
  (xpcall (lambda () (llm-context-new m '((pooling-type . bogus))))
          (lambda (e) e)))
(display "unknown pooling type raises an error") (display ": ") (write bad-pooling) (newline)

(define bad-text
  (xpcall (lambda () (llm-embed pctx 42)) (lambda (e) e)))
(display "non-string text raises an error") (display ": ") (write bad-text) (newline)

;; ---- lifetime ------------------------------------------------------------

(display "context free is idempotent") (display ": ") (write (llm-context-free plain)) (newline)

(define uaf
  (xpcall (lambda () (llm-embed plain "hello")) (lambda (e) e)))
(display "use-after-free raises an error") (display ": ") (write uaf) (newline)

(display "pooled context free is idempotent") (display ": ") (write (llm-context-free pctx)) (newline)
(display "model free is idempotent") (display ": ") (write (llm-model-free m)) (newline)

(display "done") (newline)
