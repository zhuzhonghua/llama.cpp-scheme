;; Model / vocabulary binding smoke test.
;;
;;   ./build/scmllama test/tokenize.scm

(require "test.common")

;; ---- model handle --------------------------------------------------------

(define m (llm-model-load model-path))

(display "model? ") (display ": ") (write (llm-model? m)) (newline)
(display "model? rejects non-handle") (display ": ") (write (llm-model? 42)) (newline)

(define info (llm-model-info m))
(display "model desc : ") (display (hash-table-ref info "desc")) (newline)
(display "vocab size : ") (display (hash-table-ref info "n-vocab")) (newline)
(display "n-layer    : ") (display (hash-table-ref info "n-layer")) (newline)
(display "n-embd     : ") (display (hash-table-ref info "n-embd")) (newline)

(display "model info has desc") (display ": ") (write (> (string-length (hash-table-ref info "desc")) 0)) (newline)
(display "model info has vocab size") (display ": ") (write (> (hash-table-ref info "n-vocab") 0)) (newline)

;; ---- tokenize / detokenize ----------------------------------------------

(define text "Once upon a time")
(define tokens (llm-tokenize m text '((add-special . #f))))

(display "tokens     : ") (write tokens) (newline)
(display "tokenize yields tokens") (display ": ") (write (> (vector-length tokens) 0)) (newline)

(define roundtrip (llm-detokenize m tokens '((remove-special . #f))))
(display "roundtrip  : ") (write roundtrip) (newline)
(display "detokenize(tokenize(text)) == text") (display ": ") (write roundtrip) (newline)

;; Token lists are accepted too.
(display "detokenize accepts a list") (display ": ") (write (llm-detokenize m (vector->list tokens) '((remove-special . #f)))) (newline)

;; ---- token -> piece ------------------------------------------------------

(define piece (llm-token-to-piece m (vector-ref tokens 0)))
(display "first piece: ") (write piece) (newline)
(display "token-to-piece yields text") (display ": ") (write (> (string-length piece) 0)) (newline)
(display "first piece matches text") (display ": ") (write piece) (newline)

;; ---- end of generation ---------------------------------------------------

(define eos (hash-table-ref info "eos-token"))
(display "eos token  : ") (display eos) (newline)
(display "eog? recognises eos") (display ": ") (write (llm-eog? m eos)) (newline)
(display "eog? rejects a text token") (display ": ") (write (llm-eog? m (vector-ref tokens 0))) (newline)

;; ---- lifetime ------------------------------------------------------------

(display "free is idempotent") (display ": ") (write (llm-model-free m)) (newline)
(display "free is idempotent (second call)") (display ": ") (write (llm-model-free m)) (newline)
(display "freed handle is no longer a model") (display ": ") (write (llm-model? m)) (newline)

;; Use-after-free must be a readable Scheme error, not a crash.
(define uaf (xpcall (lambda () (llm-tokenize m "x")) (lambda (e) e)))
(display "use-after-free raises an error") (display ": ") (write uaf) (newline)

;; A loading failure must also be a Scheme error.
(define missing
  (xpcall (lambda () (llm-model-load "models/does-not-exist.gguf"))
          (lambda (e) e)))
(display "missing model raises an error") (display ": ") (write missing) (newline)

(display "done") (newline)
