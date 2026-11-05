;; Model / vocabulary binding smoke test.
;;
;;   ./build/scmllama test/tokenize.scm

(load "test/common.scm")

;; ---- model handle --------------------------------------------------------

(define m (llm-model-load model-path))

(check-true "model? " (llm-model? m))
(check "model? rejects non-handle" (llm-model? 42) #f)

(define info (llm-model-info m))
(display "model desc : ") (display (hash-table-ref info "desc")) (newline)
(display "vocab size : ") (display (hash-table-ref info "n-vocab")) (newline)
(display "n-layer    : ") (display (hash-table-ref info "n-layer")) (newline)
(display "n-embd     : ") (display (hash-table-ref info "n-embd")) (newline)

(check-true "model info has desc"
            (> (string-length (hash-table-ref info "desc")) 0))
(check-true "model info has vocab size"
            (> (hash-table-ref info "n-vocab") 0))

;; ---- tokenize / detokenize ----------------------------------------------

(define text "Once upon a time")
(define tokens (llm-tokenize m text '((add-special . #f))))

(display "tokens     : ") (write tokens) (newline)
(check-true "tokenize yields tokens" (> (vector-length tokens) 0))

(define roundtrip (llm-detokenize m tokens '((remove-special . #f))))
(display "roundtrip  : ") (write roundtrip) (newline)
(check "detokenize(tokenize(text)) == text" roundtrip text)

;; Token lists are accepted too.
(check "detokenize accepts a list"
       (llm-detokenize m (vector->list tokens) '((remove-special . #f)))
       text)

;; ---- token -> piece ------------------------------------------------------

(define piece (llm-token-to-piece m (vector-ref tokens 0)))
(display "first piece: ") (write piece) (newline)
(check-true "token-to-piece yields text" (> (string-length piece) 0))
(check "first piece matches text" piece "Once")

;; ---- end of generation ---------------------------------------------------

(define eos (hash-table-ref info "eos-token"))
(display "eos token  : ") (display eos) (newline)
(check-true "eog? recognises eos" (llm-eog? m eos))
(check "eog? rejects a text token" (llm-eog? m (vector-ref tokens 0)) #f)

;; ---- lifetime ------------------------------------------------------------

(check "free is idempotent" (llm-model-free m) #t)
(check "free is idempotent (second call)" (llm-model-free m) #t)
(check "freed handle is no longer a model" (llm-model? m) #f)

;; Use-after-free must be a readable Scheme error, not a crash.
(define uaf (xpcall (lambda () (llm-tokenize m "x")) (lambda (e) e)))
(check-true "use-after-free raises an error" (string? uaf))

;; A loading failure must also be a Scheme error.
(define missing
  (xpcall (lambda () (llm-model-load "models/does-not-exist.gguf"))
          (lambda (e) e)))
(check-true "missing model raises an error" (string? missing))

(display "all model/tokenize tests passed") (newline)
