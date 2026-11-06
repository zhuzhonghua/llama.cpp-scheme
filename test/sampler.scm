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

(check "sampler? rejects non-sampler" (llm-sampler? 42) #f)
(check "sampler? rejects model" (llm-sampler? m) #f)

(define greedy (llm-sampler-new m '((greedy . #t))))
(check-true "sampler? accepts greedy sampler" (llm-sampler? greedy))

;; ---- greedy --------------------------------------------------------------

(check "greedy is deterministic"
       (llm-sampler-sample greedy ctx)
       (llm-sampler-sample greedy ctx))

(define n-vocab (hash-table-ref (llm-model-info m) "n-vocab"))
(check-true "greedy token is a valid id"
            (let ((id (llm-sampler-sample greedy ctx)))
              (and (>= id 0) (< id n-vocab))))

;; ---- seeded dist ---------------------------------------------------------

(define opts '((seed . 42) (temp . 0.7) (top-k . 40) (top-p . 0.95)))
(define s1 (llm-sampler-new m opts))
(define s2 (llm-sampler-new m opts))
(check "seeded sampling is reproducible"
       (llm-sampler-sample s1 ctx)
       (llm-sampler-sample s2 ctx))

;; ---- penalties -----------------------------------------------------------

(define sp (llm-sampler-new m '((penalty-last-n . 16) (penalty-repeat . 1.2))))
(check-true "sampler? accepts penalties" (llm-sampler? sp))
(check-true "penalty sampler samples" (>= (llm-sampler-sample sp ctx) 0))

;; ---- error paths ---------------------------------------------------------

(define bad-key
  (xpcall (lambda () (llm-sampler-new m '((bad-key . 1))))
          (lambda (e) e)))
(check-true "unknown key raises error" (string? bad-key))

(define bad-top-p
  (xpcall (lambda () (llm-sampler-new m '((top-p . 1.5))))
          (lambda (e) e)))
(check-true "top-p out of range raises error" (string? bad-top-p))

(define bad-temp
  (xpcall (lambda () (llm-sampler-new m '((temp . "hot"))))
          (lambda (e) e)))
(check-true "non-numeric temp raises error" (string? bad-temp))

;; ---- lifecycle -----------------------------------------------------------

(check "free is idempotent" (llm-sampler-free greedy) #t)
(check "free is idempotent (second call)" (llm-sampler-free greedy) #t)
(check "freed sampler is no longer a sampler" (llm-sampler? greedy) #f)

(define uaf
  (xpcall (lambda () (llm-sampler-sample greedy ctx)) (lambda (e) e)))
(check-true "use-after-free raises error" (string? uaf))

(define wrong-kind
  (xpcall (lambda () (llm-sampler-sample m ctx)) (lambda (e) e)))
(check-true "model as sampler raises error" (string? wrong-kind))

(llm-sampler-free s1)
(llm-sampler-free s2)
(llm-sampler-free sp)
(llm-context-free ctx)
(llm-model-free m)

(display "all sampler tests passed") (newline)
