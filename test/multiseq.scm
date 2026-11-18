;; Multi-sequence batch demo: two texts advance in a single decode.
;;
;;   ./build/scmllama test/multiseq.scm
;;
;; What a batch is: the unit of work handed to llama_decode — a list of tokens,
;; each with a position and the sequence(s) it belongs to.  llm-batch-from-tokens
;; builds the usual single-sequence case; to feed several sequences at once,
;; allocate one with llm-batch-new and append tokens with llm-batch-add-token.
;;
;; Each added token gets a row index (its position inside the batch), and
;; llm-sampler-sample takes that row: row = which token's logits to sample.
;; Rows are not sequence ids — with two sequences interleaved in one batch they
;; simply differ.
;;
;; Here both prompts are prefilled in one decode, then every step puts one new
;; token per sequence into one batch, so the two texts grow lockstep.  The
;; second section does the same with a single sequence, the way llm-generate
;; works internally.

(require "test.common")

(define prompt-a "Once upon a time there was a")
(define prompt-b "The best way to learn Scheme is")

(define m (llm-model-load model-path))

;; n-ctx is split across the sequences unless (kv-unified . #t):
;; 2048 total leaves 1024 tokens per sequence here.
(define ctx (llm-context-new m '((n-ctx . 2048) (n-seq-max . 2))))

;; Append `tokens` for one sequence, starting at first-pos.  Only the last
;; token of a prompt is marked for logits — that is the row we sample from.
(define (add-prompt! batch tokens first-pos seq-id)
  (let loop ((ts tokens) (pos first-pos))
    (if (null? ts)
        #t
        (begin
          (llm-batch-add-token batch (car ts) pos
                               (list (cons 'seq-id seq-id)
                                     (cons 'logits (null? (cdr ts)))))
          (loop (cdr ts) (+ pos 1))))))

(define toks-a (vector->list (llm-tokenize m prompt-a '((add-special . #f)))))
(define toks-b (vector->list (llm-tokenize m prompt-b '((add-special . #f)))))

;; One batch for both prompts: capacity = all tokens, n-seq-max = 2 seq ids
;; per token (needed for (seq-ids . (...)); seq-id alone needs just 1).
(define prefill (llm-batch-new (+ (length toks-a) (length toks-b))
                               '((n-seq-max . 2))))
(add-prompt! prefill toks-a 0 0)
(add-prompt! prefill toks-b 0 1)
(display "prefill: ") (display (llm-batch-n-tokens prefill))
(display " tokens in one decode") (newline)
(llm-decode ctx prefill)

;; One sampler per sequence: samplers keep state across tokens (penalties,
;; grammar position, RNG), so two streams must not share one.
(define smpl-a (llm-sampler-new m '((greedy . #t))))
(define smpl-b (llm-sampler-new m '((greedy . #t))))

;; Row = index of the token inside the batch.
(define row-a (- (length toks-a) 1))
(define row-b (- (+ (length toks-a) (length toks-b)) 1))
(display "row ") (display (number->string row-a)) (display " = sequence 0, row ")
(display (number->string row-b)) (display " = sequence 1") (newline)

(define next-a (llm-sampler-sample smpl-a ctx row-a))
(define next-b (llm-sampler-sample smpl-b ctx row-b))

;; Reused every step: clear drops the tokens and keeps the allocation.
(define step (llm-batch-new 2 '((n-seq-max . 2))))

(display "A: ") (display prompt-a) (newline)
(display "B: ") (display prompt-b) (newline) (newline)

(let loop ((i 0) (ta next-a) (tb next-b)
           (pos-a (length toks-a)) (pos-b (length toks-b))
           (text-a "") (text-b ""))
  (if (or (= i 12) (llm-eog? m ta) (llm-eog? m tb))
      (begin
        (newline)
        (display "A -> ") (display prompt-a) (display text-a) (newline)
        (display "B -> ") (display prompt-b) (display text-b) (newline))
      (let ((piece-a (llm-token-to-piece m ta))
            (piece-b (llm-token-to-piece m tb)))
        (display "step ") (display (number->string i))
        (display ": A ") (write piece-a)
        (display " | B ") (write piece-b) (newline)

        ;; Both new tokens go into one batch, each at its own position.
        (llm-batch-clear step)
        (llm-batch-add-token step ta pos-a
                             (list (cons 'seq-id 0) (cons 'logits #t)))
        (llm-batch-add-token step tb pos-b
                             (list (cons 'seq-id 1) (cons 'logits #t)))
        (llm-decode ctx step)

        ;; Rows of this batch: 0 = sequence 0, 1 = sequence 1.
        (loop (+ i 1)
              (llm-sampler-sample smpl-a ctx 0)
              (llm-sampler-sample smpl-b ctx 1)
              (+ pos-a 1) (+ pos-b 1)
              (string-append text-a piece-a)
              (string-append text-b piece-b)))))

(llm-batch-free prefill)
(llm-batch-free step)
(llm-sampler-free smpl-a)
(llm-sampler-free smpl-b)
(llm-context-free ctx)

;; ---- the same batch API with a single sequence -------------------------
;;
;; One sequence means: omit seq-id (it defaults to 0), count positions up from
;; 0, and mark only the last prompt token for logits.  This is what
;; llm-generate does internally; doing it by hand shows the plumbing.

(define ctx1 (llm-context-new m '((n-ctx . 1024))))  ; n-seq-max defaults to 1
(define smpl (llm-sampler-new m '((greedy . #t))))
(define batch (llm-batch-new 64))
(define prompt-c "The weather today is")
(define toks-c (vector->list (llm-tokenize m prompt-c '((add-special . #f)))))

;; Prefill the whole prompt in one decode.
(let loop ((ts toks-c) (pos 0))
  (if (null? ts)
      #t
      (begin
        (llm-batch-add-token batch (car ts) pos
                             (list (cons 'logits (null? (cdr ts)))))
        (loop (cdr ts) (+ pos 1)))))
(llm-decode ctx1 batch)

(display "single sequence: ") (display prompt-c)

;; No row argument: -1 means "last token decoded", the single-sequence case.
(let loop ((i 0) (pos (length toks-c)) (tok (llm-sampler-sample smpl ctx1)))
  (if (or (= i 12) (llm-eog? m tok))
      (newline)
      (begin
        (display (llm-token-to-piece m tok))
        ;; Feed the sampled token back: one new token, next position, seq 0.
        (llm-batch-clear batch)
        (llm-batch-add-token batch tok pos (list (cons 'logits #t)))
        (llm-decode ctx1 batch)
        (loop (+ i 1) (+ pos 1) (llm-sampler-sample smpl ctx1)))))

(llm-batch-free batch)
(llm-sampler-free smpl)
(llm-context-free ctx1)
(llm-model-free m)
