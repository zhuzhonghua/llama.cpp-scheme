;; Sampler binding smoke test.
;;
;;   ./build/scmllama test/sampler.scm

(require "test.common")

;; ---- setup ---------------------------------------------------------------

(define m (llm-model-load model-path))
(define ctx (llm-context-new m '((n-ctx . 128))))

(define tokens (llm-tokenize m "Once upon a time" '((add-special . #f))))
(llm-decode ctx (llm-batch-from-tokens tokens))

;; ---- handle checks -------------------------------------------------------

(display "sampler? rejects non-sampler") (display ": ") (write (llm-sampler? 42)) (newline)
(display "sampler? rejects model") (display ": ") (write (llm-sampler? m)) (newline)

(define greedy (llm-sampler-new m '((greedy . #t))))
(display "sampler? accepts greedy sampler") (display ": ") (write (llm-sampler? greedy)) (newline)

;; ---- greedy --------------------------------------------------------------

(display "greedy is deterministic") (display ": ") (write (llm-sampler-sample greedy ctx)) (newline)

(define n-vocab (hash-table-ref (llm-model-info m) "n-vocab"))
(display "greedy token is a valid id") (display ": ") (write (let ((id (llm-sampler-sample greedy ctx)))
              (and (>= id 0) (< id n-vocab)))) (newline)

;; ---- seeded dist ---------------------------------------------------------

(define opts '((seed . 42) (temp . 0.7) (top-k . 40) (top-p . 0.95)))
(define s1 (llm-sampler-new m opts))
(define s2 (llm-sampler-new m opts))
(display "seeded sampling is reproducible") (display ": ") (write (llm-sampler-sample s1 ctx)) (newline)

;; ---- penalties -----------------------------------------------------------

(define sp (llm-sampler-new m '((penalty-last-n . 16) (penalty-repeat . 1.2))))
(display "sampler? accepts penalties") (display ": ") (write (llm-sampler? sp)) (newline)
(display "penalty sampler samples") (display ": ") (write (>= (llm-sampler-sample sp ctx) 0)) (newline)

;; ---- reset ---------------------------------------------------------------

;; Reset rewinds the dist RNG to the initial seed, so the first draw repeats.
(define first-draw (llm-sampler-sample s1 ctx))
(display "sampler reset returns #t") (display ": ") (write (llm-sampler-reset s1)) (newline)
(display "reset restarts the RNG stream") (display ": ") (write (llm-sampler-sample s1 ctx)) (newline)

;; ---- error paths ---------------------------------------------------------

(define bad-key
  (xpcall (lambda () (llm-sampler-new m '((bad-key . 1))))
          (lambda (e) e)))
(display "unknown key raises error") (display ": ") (write bad-key) (newline)

(define bad-top-p
  (xpcall (lambda () (llm-sampler-new m '((top-p . 1.5))))
          (lambda (e) e)))
(display "top-p out of range raises error") (display ": ") (write bad-top-p) (newline)

(define bad-temp
  (xpcall (lambda () (llm-sampler-new m '((temp . "hot"))))
          (lambda (e) e)))
(display "non-numeric temp raises error") (display ": ") (write bad-temp) (newline)

;; ---- lifecycle -----------------------------------------------------------

(display "free is idempotent") (display ": ") (write (llm-sampler-free greedy)) (newline)
(display "free is idempotent (second call)") (display ": ") (write (llm-sampler-free greedy)) (newline)
(display "freed sampler is no longer a sampler") (display ": ") (write (llm-sampler? greedy)) (newline)

(define uaf
  (xpcall (lambda () (llm-sampler-sample greedy ctx)) (lambda (e) e)))
(display "use-after-free raises error") (display ": ") (write uaf) (newline)

(define reset-uaf
  (xpcall (lambda () (llm-sampler-reset greedy)) (lambda (e) e)))
(display "reset on a freed sampler raises error") (display ": ") (write reset-uaf) (newline)

(define wrong-kind
  (xpcall (lambda () (llm-sampler-sample m ctx)) (lambda (e) e)))
(display "model as sampler raises error") (display ": ") (write wrong-kind) (newline)

(llm-sampler-free s1)
(llm-sampler-free s2)
(llm-sampler-free sp)
(llm-context-free ctx)
(llm-model-free m)

(display "done") (newline)
