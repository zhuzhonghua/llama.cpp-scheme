;; GGUF metadata demo.
;;
;;   ./build/scmllama test/model-meta.scm
;;
;; The backend silences llama.cpp/ggml logging before startup, so the Metal
;; device init and model load messages do not appear.

(define m (llm-model-load "models/stories260K.gguf"))
(define meta (llm-model-meta m))

;; A hash-table prints as #<hash-table>, so walk it.  Order follows the table's
;; internal layout; values are strings (numbers and booleans are rendered as
;; text), and array metadata such as tokenizer.ggml.tokens is not included.
(display "GGUF metadata:") (newline)
(hash-table-for-each
  (lambda (key value)
    (display "  ")
    (display key)
    (display " = ")
    (write value)
    (newline))
  meta)
(newline)

;; Single key lookup.
(display "architecture: ")
(write (hash-table-ref meta "general.architecture"))
(newline)

(llm-model-free m)
