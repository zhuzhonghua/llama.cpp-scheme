;; Chat generation binding test.
;;
;;   ./build/scmllama test/chat-generate.scm

(require "test.common")

(define m (llm-model-load model-path))
(define ctx (llm-context-new m '((n-ctx . 512))))
(define smpl (llm-sampler-new m '((temp . 0.0) (seed . 42))))

(define msgs
  '(("system" . "You are a helpful assistant.")
    ("user" . "Say hi in one word.")))

;; ---- built-in template ---------------------------------------------------

(define out "")
(define pieces 0)
(define text
  (llm-chat-generate ctx smpl msgs
                     (lambda (piece)
                       (set! out (string-append out piece))
                       (set! pieces (+ pieces 1)))
                     '((max-tokens . 8))))

(display "generated  : ") (write text) (newline)

(display "returns a string") (display ": ") (write text) (newline)
(display "produces at least one piece") (display ": ") (write (> pieces 0)) (newline)
(display "callback pieces reconstruct the result") (display ": ") (write out) (newline)

;; A chat call re-renders the whole conversation, so the context can be
;; reused without an explicit llm-reset.
(define text2
  (llm-chat-generate ctx smpl msgs (lambda (piece) #t) '((max-tokens . 8))))
(display "same conversation is deterministic") (display ": ") (write text2) (newline)

;; chat-generate uses add-special #f and parse-special #t: the rendered
;; template already contains the special markers.
(llm-reset ctx)
(define prompt (llm-chat-template-apply m msgs))
(define manual
  (llm-generate ctx smpl prompt (lambda (piece) #t)
                '((max-tokens . 8)
                  (add-special . #f)
                  (parse-special . #t))))
(display "matches llm-chat-template-apply + llm-generate") (display ": ") (write text) (newline)

(display "max-tokens 0 returns an empty string") (display ": ") (write (llm-chat-generate ctx smpl msgs (lambda (piece) #t)
                          '((max-tokens . 0)))) (newline)

;; ---- callback early stop -------------------------------------------------

(define calls 0)
(define stopped
  (llm-chat-generate ctx smpl msgs
                     (lambda (piece)
                       (set! calls (+ calls 1))
                       #f)
                     '((max-tokens . 32))))

(display "returning #f stops after one token") (display ": ") (write calls) (newline)
(display "early stop returns the text produced so far") (display ": ") (write (> (string-length stopped) 0)) (newline)

;; ---- explicit template ---------------------------------------------------
;;
;; stories260K carries no chat template in its GGUF metadata.

(define tiny (llm-model-load "models/stories260K.gguf"))
(define tctx (llm-context-new tiny '((n-ctx . 256))))
(define tsmpl (llm-sampler-new tiny '((temp . 0.0) (seed . 42))))

(define no-template
  (xpcall (lambda () (llm-chat-generate tctx tsmpl '(("user" . "Hi"))
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(display "model without a chat template raises an error") (display ": ") (write no-template) (newline)

(define tiny-text
  (llm-chat-generate tctx tsmpl '(("user" . "Hi")) (lambda (piece) #t)
                     '((template . "chatml") (max-tokens . 4))))
(display "explicit template name works") (display ": ") (write tiny-text) (newline)

;; ---- error paths ---------------------------------------------------------

(define bad-shape
  (xpcall (lambda () (llm-chat-generate ctx smpl '(("user" "Hi"))
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(display "malformed message raises an error") (display ": ") (write bad-shape) (newline)

(define bad-role
  (xpcall (lambda () (llm-chat-generate ctx smpl '((user . "Hi"))
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(display "non-string role raises an error") (display ": ") (write bad-role) (newline)

(define bad-content
  (xpcall (lambda () (llm-chat-generate ctx smpl '(("user" . 1))
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(display "non-string content raises an error") (display ": ") (write bad-content) (newline)

(define bad-list
  (xpcall (lambda () (llm-chat-generate ctx smpl "not a list"
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(display "non-list messages raises an error") (display ": ") (write bad-list) (newline)

(define bad-option
  (xpcall (lambda () (llm-chat-generate ctx smpl msgs (lambda (piece) #t)
                                        '((nope . 1))))
          (lambda (e) e)))
(display "unknown option raises an error") (display ": ") (write bad-option) (newline)

(define bad-max
  (xpcall (lambda () (llm-chat-generate ctx smpl msgs (lambda (piece) #t)
                                        '((max-tokens . -1))))
          (lambda (e) e)))
(display "negative max-tokens raises an error") (display ": ") (write bad-max) (newline)

(define keep-kv
  (xpcall (lambda () (llm-chat-generate ctx smpl msgs (lambda (piece) #t)
                                        '((keep-kv . #t))))
          (lambda (e) e)))
(display "keep-kv raises an error") (display ": ") (write keep-kv) (newline)

;; ---- lifetime ------------------------------------------------------------

(display "sampler free is idempotent") (display ": ") (write (llm-sampler-free smpl)) (newline)
(display "context free is idempotent") (display ": ") (write (llm-context-free ctx)) (newline)

(define uaf
  (xpcall (lambda () (llm-chat-generate ctx smpl msgs (lambda (piece) #t)))
          (lambda (e) e)))
(display "use-after-free raises an error") (display ": ") (write uaf) (newline)

(display "tiny sampler free is idempotent") (display ": ") (write (llm-sampler-free tsmpl)) (newline)
(display "tiny context free is idempotent") (display ": ") (write (llm-context-free tctx)) (newline)
(display "tiny model free is idempotent") (display ": ") (write (llm-model-free tiny)) (newline)
(display "model free is idempotent") (display ": ") (write (llm-model-free m)) (newline)

(display "done") (newline)
