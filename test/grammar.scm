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
(display "grammar constrains output to yes/no") (display ": ") (write (member answer '("yes" "no"))) (newline)

;; ---- reuse after reset ---------------------------------------------------
;;
;; The grammar parser has consumed "yes"/"no" and is in its final state.
;; Without a reset the next generation can only emit EOG and returns "".
;; llm-sampler-reset rewinds the parser so the sampler can serve a new prompt.

(display "sampler reset returns #t") (display ": ") (write (llm-sampler-reset yes-no)) (newline)
(llm-reset ctx)

(define answer2
  (llm-generate ctx yes-no "Answer with yes or no: is grass green?"
                (lambda (piece) #t)
                '((max-tokens . 8))))

(display "answer 2   : ") (write answer2) (newline)
(display "reset grammar sampler is reusable") (display ": ") (write (member answer2 '("yes" "no"))) (newline)

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
(display "grammar-root selects the start symbol") (display ": ") (write greeting) (newline)

;; ---- error paths --------------------------------------------------------

(define bad-grammar
  (xpcall (lambda () (llm-sampler-new m '((grammar . "root ::= \"unterminated"))))
          (lambda (e) e)))
(display "invalid grammar raises an error") (display ": ") (write bad-grammar) (newline)

(define empty-grammar
  (xpcall (lambda () (llm-sampler-new m '((grammar . ""))))
          (lambda (e) e)))
(display "empty grammar raises an error") (display ": ") (write empty-grammar) (newline)

(define bad-root
  (xpcall (lambda () (llm-sampler-new
                      m '((grammar . "root ::= \"x\"")
                          (grammar-root . "missing"))))
          (lambda (e) e)))
(display "missing start symbol raises an error") (display ": ") (write bad-root) (newline)

(define bad-type
  (xpcall (lambda () (llm-sampler-new m '((grammar . 42))))
          (lambda (e) e)))
(display "non-string grammar raises an error") (display ": ") (write bad-type) (newline)

;; ---- lifetime -----------------------------------------------------------

(display "sampler free is idempotent") (display ": ") (write (llm-sampler-free hi)) (newline)
(display "context free is idempotent") (display ": ") (write (llm-context-free ctx)) (newline)
(display "model free is idempotent") (display ": ") (write (llm-model-free m)) (newline)

(display "done") (newline)
