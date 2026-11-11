;; GBNF grammar constrained sampling.
;;
;;   ./build/scmllama test/grammar.scm

(require "test.common")

(define m (llm-model-load model-path))
(define ctx (llm-context-new m '((n-ctx . 512))))

;; ---- yes / no ------------------------------------------------------------
;;
;; Greedy decoding + grammar: whatever the model would prefer, only "yes" and
;; "no" can be sampled.

(define yes-no
  (llm-sampler-new m '((greedy . #t)
                       (grammar . "root ::= \"yes\" | \"no\""))))

(define answer
  (llm-generate ctx yes-no "Answer with yes or no: is the sky blue?"
                (lambda (piece) #t)
                '((max-tokens . 8))))

(display "answer     : ") (write answer) (newline)
(check-true "grammar constrains output to yes/no"
            (member answer '("yes" "no")))

;; ---- custom start symbol ------------------------------------------------

(llm-reset ctx)
(llm-sampler-free yes-no)

(define hi
  (llm-sampler-new m '((greedy . #t)
                       (grammar . "greeting ::= \"hi\"")
                       (grammar-root . "greeting"))))

(define greeting
  (llm-generate ctx hi "Ignore the model; say hello."
                (lambda (piece) #t)
                '((max-tokens . 8))))

(display "greeting   : ") (write greeting) (newline)
(check "grammar-root selects the start symbol" greeting "hi")

;; ---- error paths --------------------------------------------------------

(define bad-grammar
  (xpcall (lambda () (llm-sampler-new m '((grammar . "root ::= \"unterminated"))))
          (lambda (e) e)))
(check-true "invalid grammar raises an error" (string? bad-grammar))

(define empty-grammar
  (xpcall (lambda () (llm-sampler-new m '((grammar . ""))))
          (lambda (e) e)))
(check-true "empty grammar raises an error" (string? empty-grammar))

(define bad-root
  (xpcall (lambda () (llm-sampler-new
                      m '((grammar . "root ::= \"x\"")
                          (grammar-root . "missing"))))
          (lambda (e) e)))
(check-true "missing start symbol raises an error" (string? bad-root))

(define bad-type
  (xpcall (lambda () (llm-sampler-new m '((grammar . 42))))
          (lambda (e) e)))
(check-true "non-string grammar raises an error" (string? bad-type))

;; ---- lifetime -----------------------------------------------------------

(check "sampler free is idempotent" (llm-sampler-free hi) #t)
(check "context free is idempotent" (llm-context-free ctx) #t)
(check "model free is idempotent" (llm-model-free m) #t)

(display "all grammar tests passed") (newline)
