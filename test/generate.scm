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
(check-true "returns a string" (string? out1))
(check-true "output is non-empty" (> (string-length out1) 0))

;; ---- callback receives the same pieces as the returned text --------------

(llm-reset ctx)
(define acc "")
(define out2
  (llm-generate ctx smpl prompt
                (lambda (piece) (set! acc (string-append acc piece)))
                '((max-tokens . 8))))
(check "callback pieces concatenate to the result" acc out2)

;; ---- determinism ---------------------------------------------------------

(llm-reset ctx)
(define out3
  (llm-generate ctx smpl prompt (lambda (piece) #t) '((max-tokens . 8))))
(check "greedy generation is deterministic" out3 out2)

;; ---- max-tokens ----------------------------------------------------------

(llm-reset ctx)
(define short-len
  (string-length (llm-generate ctx smpl prompt (lambda (p) #t) '((max-tokens . 4)))))
(llm-reset ctx)
(define long-len
  (string-length (llm-generate ctx smpl prompt (lambda (p) #t) '((max-tokens . 8)))))
(check-true "max-tokens bounds the length" (<= short-len long-len))

;; ---- error paths ---------------------------------------------------------

(llm-reset ctx)
(define empty-err
  (xpcall (lambda ()
            (llm-generate ctx smpl "" (lambda (p) #t) '((add-special . #f))))
          (lambda (e) e)))
(check-true "empty prompt raises error" (string? empty-err))

(define bad-key
  (xpcall (lambda ()
            (llm-generate ctx smpl prompt (lambda (p) #t) '((bad-key . 1))))
          (lambda (e) e)))
(check-true "unknown option raises error" (string? bad-key))

(define cb-err
  (xpcall (lambda ()
            (llm-generate ctx smpl prompt (lambda (p) (error "boom"))
                          '((max-tokens . 2))))
          (lambda (e) e)))
(check-true "callback error propagates" (string? cb-err))

(define wrong-kind
  (xpcall (lambda ()
            (llm-generate m smpl prompt (lambda (p) #t)))
          (lambda (e) e)))
(check-true "model as context raises error" (string? wrong-kind))

;; ---- cleanup -------------------------------------------------------------

(llm-sampler-free smpl)
(llm-context-free ctx)
(llm-model-free m)

(display "all generate tests passed") (newline)
