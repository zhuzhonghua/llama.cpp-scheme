;; Single-sequence batch demo: step-by-step generation with the low-level API.
;;
;;   ./build/scmllama test/batch-single-seq.scm
;;
;; This does by hand what llm-generate does internally: prefill a prompt, then
;; repeatedly decode one sampled token.  Only seq 0 is involved, so:
;;   - llm-batch-add-token needs no seq-id / seq-ids (they default to seq 0)
;;   - llm-sampler-sample needs no row (it defaults to -1 = the last token
;;     decoded, i.e. the single sequence's only output position)
;; Those knobs exist only for the multi-sequence case; see multiseq.scm.
;;
;; The loop:
;;   1. Tokenize the prompt, feed every token into a batch (positions 0..n-1),
;;      marking only the last one for logits, and decode once to prefill the KV.
;;   2. Sample the first token from that last position.
;;   3. Each step: print the token, clear the batch, put that single token back
;;      at the next position with logits on, decode, and sample the next one.
;;   4. Stop at EOG or after max-steps tokens.
;;
;; One reusable batch serves both phases: llm-batch-clear keeps its allocation.

(require "test.common")

(define max-steps 16)
(define prompt "Once upon a time")

(define m (llm-model-load model-path))

;; n-seq-max defaults to 1: a single sequence, which is all this demo uses.
(define ctx (llm-context-new m '((n-ctx . 512))))
(define smpl (llm-sampler-new m '((greedy . #t))))

(define toks (vector->list (llm-tokenize m prompt '((add-special . #f)))))

;; Capacity covers the whole prefill; the same batch is cleared and reused for
;; the one-token-per-step loop below.
(define batch (llm-batch-new (length toks)))

;; ---- phase 1: prefill -----------------------------------------------------
;; Append every prompt token; only the last gets (logits . #t), because that is
;; the position we sample from.  No seq-id option -> every token is seq 0.
(let loop ((ts toks) (pos 0))
  (if (null? ts)
      #t
      (begin
        (llm-batch-add-token batch (car ts) pos
                             (list (cons 'logits (null? (cdr ts)))))
        (loop (cdr ts) (+ pos 1)))))
(llm-decode ctx batch)

(display "prompt: ") (display prompt) (newline)
(display (number->string (llm-batch-n-tokens batch)))
(display " prompt tokens prefilled in one decode") (newline)
(display "generated: ")

;; No row argument: -1 means "the last token decoded" - the single-seq case.
(define next (llm-sampler-sample smpl ctx))

;; ---- phase 2: step-by-step generation ------------------------------------
(let loop ((i 0) (pos (length toks)) (tok next))
  (if (or (= i max-steps) (llm-eog? m tok))
      (begin (newline) (newline))
      (begin
        (display (llm-token-to-piece m tok))
        ;; Feed the sampled token back as a one-token batch at the next position.
        (llm-batch-clear batch)
        (llm-batch-add-token batch tok pos (list (cons 'logits #t)))
        (llm-decode ctx batch)
        (loop (+ i 1) (+ pos 1) (llm-sampler-sample smpl ctx)))))

(llm-batch-free batch)
(llm-sampler-free smpl)
(llm-context-free ctx)
(llm-model-free m)
