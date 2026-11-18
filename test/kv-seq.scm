;; KV sequence copy / keep demo: fork a cached prefix, then prune it.
;;
;;   ./build/scmllama test/kv-seq.scm
;;
;; llm-kv-copy and llm-kv-keep wrap llama_memory_seq_cp / seq_keep.  They are
;; the two operations needed for beam search over a multi-sequence context:
;;   - copy  duplicates one sequence's KV into another so both share the same
;;           prefix and can then diverge (beam expand / fork).
;;   - keep  drops every sequence except one (beam prune).
;;
;; Two things to watch:
;;   1. src/dst are sequence ids (0 .. n-seq-max-1), NOT batch row indices.
;;      A row is which token in the batch to sample from; a seq id is which
;;      sequence's KV a token belongs to.
;;   2. keep needs (kv-unified . #t).  With the default kv-unified=#f each
;;      sequence lives in its own KV "stream", and seq_keep only touches the
;;      kept sequence's stream -- it cannot remove another sequence's cells.
;;      Unified KV puts all sequences in one stream so keep can prune across
;;      them.
;;
;; This is a demonstration script (no assertions): run it and read the output.

(require "test.common")

(define m (llm-model-load model-path))

;; =========================================================================
;; 1. copy: fork seq 0's prefix into seq 1
;; =========================================================================
(define ctx (llm-context-new m '((n-ctx . 512) (n-batch . 256) (n-seq-max . 2))))
(define smpl (llm-sampler-new m '((greedy . #t))))

(define prompt "Once upon a time there was a")
(define toks (vector->list (llm-tokenize m prompt '((add-special . #f)))))
(define n (length toks))

;; Prefill the prompt into seq 0 only.  Only the last token needs logits;
;; that is the row we sample the first generated token from.
(define prefill (llm-batch-new n '((n-seq-max . 2))))
(let loop ((ts toks) (pos 0))
  (if (null? ts)
      #t
      (begin
        (llm-batch-add-token prefill (car ts) pos
                             (list (cons 'seq-id 0)
                                   (cons 'logits (null? (cdr ts)))))
        (loop (cdr ts) (+ pos 1)))))
(llm-decode ctx prefill)
(define first (llm-sampler-sample smpl ctx (- n 1)))

;; Copy the whole prefix to seq 1.  This ctx is non-unified, so src and dst
;; live in different KV streams and llama.cpp only supports copying the FULL
;; sequence (p0/p1 must be -1 or 0): passing a middle range aborts the process
;; with "seq_cp() is only supported for full KV buffers".  Arbitrary ranges
;; work on a (kv-unified . #t) context, where all sequences share one stream.
;; The optional form is therefore (llm-kv-copy ctx 0 1 0 -1) here, not (0 n).
(display "copy seq 0 -> seq 1: ") (display (llm-kv-copy ctx 0 1)) (newline)

;; Now feed the SAME next token into both sequences at the same position,
;; in one batch.  Because seq 1 inherited seq 0's KV, both rows produce the
;; same logits and greedy sampling continues identically.
(define step (llm-batch-new 2 '((n-seq-max . 2))))

(define (forked-text)
  (let loop ((i 0) (pos n) (tok first) (acc " * "))
    (if (or (= i 12) (llm-eog? m tok))
        acc
        (let ((piece (llm-token-to-piece m tok)))
          (llm-batch-clear step)
          (llm-batch-add-token step tok pos
                               (list (cons 'seq-id 0) (cons 'logits #t)))
          (llm-batch-add-token step tok pos
                               (list (cons 'seq-id 1) (cons 'logits #t)))
          (llm-decode ctx step)
          ;; row 0 -> seq 0's token, row 1 -> seq 1's token
          (loop (+ i 1) (+ pos 1)
                (llm-sampler-sample smpl ctx 0)
                (string-append acc piece))))))

(display prompt) (newline)
(display "seq 0 and seq 1 (identical prefix): ") (display (forked-text)) (newline)

(llm-batch-free prefill)
(llm-batch-free step)
(llm-sampler-free smpl)
(llm-context-free ctx)

;; =========================================================================
;; 2. keep: prune all sequences except one
;; =========================================================================
;; kv-unified=#t is what makes keep able to see (and drop) other sequences.
(define ctx2 (llm-context-new m '((n-ctx . 512) (n-batch . 256)
                                  (n-seq-max . 2) (kv-unified . #t))))
(define smpl2 (llm-sampler-new m '((greedy . #t))))

(define pa "Once upon a time there was a")
(define pb "The quick brown fox jumps over the lazy dog")
(define ta (vector->list (llm-tokenize m pa '((add-special . #f)))))
(define tb (vector->list (llm-tokenize m pb '((add-special . #f)))))
(define la (length ta))
(define lb (length tb))

;; Two independent prefixes in one batch: ta -> seq 0, tb -> seq 1.
(define both (llm-batch-new (+ la lb) '((n-seq-max . 2))))
(let loop ((ts ta) (pos 0))
  (if (null? ts)
      #t
      (begin
        (llm-batch-add-token both (car ts) pos
                             (list (cons 'seq-id 0)
                                   (cons 'logits (null? (cdr ts)))))
        (loop (cdr ts) (+ pos 1)))))
(let loop ((ts tb) (pos 0))
  (if (null? ts)
      #t
      (begin
        (llm-batch-add-token both (car ts) pos
                             (list (cons 'seq-id 1)
                                   (cons 'logits (null? (cdr ts)))))
        (loop (cdr ts) (+ pos 1)))))
(llm-decode ctx2 both)

(display "before keep: seq 0 / seq 1 KV tokens = ")
(display (llm-kv-pos ctx2 0)) (display " / ") (display (llm-kv-pos ctx2 1))
(newline)
(display "keep seq 1: ") (display (llm-kv-keep ctx2 1)) (newline)
(display "after keep:  seq 0 / seq 1 KV tokens = ")
(display (llm-kv-pos ctx2 0)) (display " / ") (display (llm-kv-pos ctx2 1))
(newline)

;; Sampling is by BATCH ROW, not by seq id.  `both` held 16 tokens: ta in
;; rows 0..la-1, then tb in rows la..la+lb-1.  Only tb's last token carries
;; logits, and it sits at row la+lb-1 -- so that is the row we sample to get
;; seq 1's next token.
(define next-token (llm-sampler-sample smpl2 ctx2 (+ la lb -1)))
;; That token is then decoded in a NEW, single-token batch `cont`.  Its
;; logits live in row 0 of `cont`, so the follow-up sample uses row 0 here,
;; not la+lb-1 (which belonged to the previous batch).
(define cont (llm-batch-new 2 '((n-seq-max . 2))))
(llm-batch-add-token cont next-token lb '((seq-id . 1) (logits . #t)))
(llm-decode ctx2 cont)
(display "kept seq 1 continues: ")
(display pb) (display (llm-token-to-piece m (llm-sampler-sample smpl2 ctx2 0)))
(newline)

(llm-batch-free both)
(llm-batch-free cont)
(llm-sampler-free smpl2)
(llm-context-free ctx2)
(llm-model-free m)

(display "done") (newline)
