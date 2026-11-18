;; Generation loop smoke test.
;;
;;   ./build/scmllama test/generate.scm

(require "test.common")

(define m (llm-model-load model-path))
(define ctx (llm-context-new m '((n-ctx . 512))))

;; greedy sampling makes the output fully deterministic
(define smpl (llm-sampler-new m '((greedy . #t))))
(define prompt "Once upon a time")

;; ---- shape ---------------------------------------------------------------

(define out1 (llm-generate ctx smpl prompt (lambda (piece) #t) '((max-tokens . 8))))
(display "returns a string") (display ": ") (write out1) (newline)
(display "output is non-empty") (display ": ") (write (> (string-length out1) 0)) (newline)

;; ---- callback receives the same pieces as the returned text --------------

(llm-reset ctx)
(define acc "")
(define out2
  (llm-generate ctx smpl prompt
                (lambda (piece) (set! acc (string-append acc piece)))
                '((max-tokens . 8))))
(display "callback pieces concatenate to the result") (display ": ") (write acc) (newline)

;; ---- determinism ---------------------------------------------------------

(llm-reset ctx)
(define out3
  (llm-generate ctx smpl prompt (lambda (piece) #t) '((max-tokens . 8))))
(display "greedy generation is deterministic") (display ": ") (write out3) (newline)

;; ---- max-tokens ----------------------------------------------------------

(llm-reset ctx)
(define short-len
  (string-length (llm-generate ctx smpl prompt (lambda (p) #t) '((max-tokens . 4)))))
(llm-reset ctx)
(define long-len
  (string-length (llm-generate ctx smpl prompt (lambda (p) #t) '((max-tokens . 8)))))
(display "max-tokens bounds the length") (display ": ") (write (<= short-len long-len)) (newline)

;; ---- early stop via the callback's return value ---------------------------

;; Returning #f from the callback ends generation at this token; any other
;; value keeps it going.
(llm-reset ctx)
(define seen 0)
(define partial
  (llm-generate ctx smpl prompt
                (lambda (piece)
                  (set! seen (+ seen 1))
                  (if (= seen 3) #f #t))
                '((max-tokens . 40))))
(display "callback #f stops generation") (display ": ") (write seen) (newline)
(display "partial text is returned") (display ": ") (write (> (string-length partial) 0)) (newline)

;; Nothing to reset afterwards: stopping is plain control flow.
(llm-reset ctx)
(define after-stop
  (llm-generate ctx smpl prompt (lambda (p) #t) '((max-tokens . 4))))
(display "generation works after an early stop") (display ": ") (write (string-length after-stop)) (newline)

;; ---- error paths ---------------------------------------------------------

(llm-reset ctx)
(define empty-err
  (xpcall (lambda ()
            (llm-generate ctx smpl "" (lambda (p) #t) '((add-special . #f))))
          (lambda (e) e)))
(display "empty prompt raises error") (display ": ") (write empty-err) (newline)

(define bad-key
  (xpcall (lambda ()
            (llm-generate ctx smpl prompt (lambda (p) #t) '((bad-key . 1))))
          (lambda (e) e)))
(display "unknown option raises error") (display ": ") (write bad-key) (newline)

(define cb-err
  (xpcall (lambda ()
            (llm-generate ctx smpl prompt (lambda (p) (error "boom"))
                          '((max-tokens . 2))))
          (lambda (e) e)))
(display "callback error propagates") (display ": ") (write cb-err) (newline)

(define wrong-kind
  (xpcall (lambda ()
            (llm-generate m smpl prompt (lambda (p) #t)))
          (lambda (e) e)))
(display "model as context raises error") (display ": ") (write wrong-kind) (newline)

;; ---- cleanup -------------------------------------------------------------

(llm-sampler-free smpl)
(llm-context-free ctx)
(llm-model-free m)

(display "done") (newline)
