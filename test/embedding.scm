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

(check "embedding length is n-embd" (vector-length a) n-embd)
(check-true "embedding components are real numbers" (real? (vector-ref a 0)))

(define a2 (llm-embed ctx "Once upon a time"))
(check "same text yields the same vector" (equal? a a2) #t)

(define b (llm-embed ctx "The quick brown fox"))
(check "different text yields a different vector" (equal? a b) #f)

;; ---- mean pooling: one vector per sequence -------------------------------

(llm-context-free ctx)

(define pctx
  (llm-context-new m '((n-ctx . 512)
                       (embeddings . #t)
                       (pooling-type . mean))))

(define pooled (llm-embed pctx "Once upon a time"))
(display "dim (mean) : ") (display (vector-length pooled)) (newline)

(check "mean-pooled embedding length is n-embd"
       (vector-length pooled) n-embd)
(check-true "mean pooling differs from the last token"
            (not (equal? pooled a)))

;; ---- error paths ---------------------------------------------------------

(define plain (llm-context-new m '((n-ctx . 256))))

(define no-embeddings
  (xpcall (lambda () (llm-embed plain "hello")) (lambda (e) e)))
(check-true "context without embeddings raises an error"
            (string? no-embeddings))

;; Empty text has no tokens to decode.
(define empty-text
  (xpcall (lambda () (llm-embed pctx "" '((add-special . #f))))
          (lambda (e) e)))
(check-true "empty text raises an error" (string? empty-text))

(define bad-pooling
  (xpcall (lambda () (llm-context-new m '((pooling-type . bogus))))
          (lambda (e) e)))
(check-true "unknown pooling type raises an error" (string? bad-pooling))

(define bad-text
  (xpcall (lambda () (llm-embed pctx 42)) (lambda (e) e)))
(check-true "non-string text raises an error" (string? bad-text))

;; ---- lifetime ------------------------------------------------------------

(check "context free is idempotent" (llm-context-free plain) #t)

(define uaf
  (xpcall (lambda () (llm-embed plain "hello")) (lambda (e) e)))
(check-true "use-after-free raises an error" (string? uaf))

(check "pooled context free is idempotent" (llm-context-free pctx) #t)
(check "model free is idempotent" (llm-model-free m) #t)

(display "all embedding tests passed") (newline)
