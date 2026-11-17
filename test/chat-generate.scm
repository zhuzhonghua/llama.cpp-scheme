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

(check-true "returns a string" (string? text))
(check-true "produces at least one piece" (> pieces 0))
(check "callback pieces reconstruct the result" out text)

;; A chat call re-renders the whole conversation, so the context can be
;; reused without an explicit llm-reset.
(define text2
  (llm-chat-generate ctx smpl msgs (lambda (piece) #t) '((max-tokens . 8))))
(check "same conversation is deterministic" text2 text)

;; chat-generate uses add-special #f and parse-special #t: the rendered
;; template already contains the special markers.
(llm-reset ctx)
(define prompt (llm-chat-template-apply m msgs))
(define manual
  (llm-generate ctx smpl prompt (lambda (piece) #t)
                '((max-tokens . 8)
                  (add-special . #f)
                  (parse-special . #t))))
(check "matches llm-chat-template-apply + llm-generate" text manual)

(check "max-tokens 0 returns an empty string"
       (llm-chat-generate ctx smpl msgs (lambda (piece) #t)
                          '((max-tokens . 0)))
       "")

;; ---- callback early stop -------------------------------------------------

(define calls 0)
(define stopped
  (llm-chat-generate ctx smpl msgs
                     (lambda (piece)
                       (set! calls (+ calls 1))
                       #f)
                     '((max-tokens . 32))))

(check "returning #f stops after one token" calls 1)
(check-true "early stop returns the text produced so far"
            (> (string-length stopped) 0))

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
(check-true "model without a chat template raises an error"
            (string? no-template))

(define tiny-text
  (llm-chat-generate tctx tsmpl '(("user" . "Hi")) (lambda (piece) #t)
                     '((template . "chatml") (max-tokens . 4))))
(check-true "explicit template name works" (string? tiny-text))

;; ---- error paths ---------------------------------------------------------

(define bad-shape
  (xpcall (lambda () (llm-chat-generate ctx smpl '(("user" "Hi"))
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(check-true "malformed message raises an error" (string? bad-shape))

(define bad-role
  (xpcall (lambda () (llm-chat-generate ctx smpl '((user . "Hi"))
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(check-true "non-string role raises an error" (string? bad-role))

(define bad-content
  (xpcall (lambda () (llm-chat-generate ctx smpl '(("user" . 1))
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(check-true "non-string content raises an error" (string? bad-content))

(define bad-list
  (xpcall (lambda () (llm-chat-generate ctx smpl "not a list"
                                        (lambda (piece) #t)))
          (lambda (e) e)))
(check-true "non-list messages raises an error" (string? bad-list))

(define bad-option
  (xpcall (lambda () (llm-chat-generate ctx smpl msgs (lambda (piece) #t)
                                        '((nope . 1))))
          (lambda (e) e)))
(check-true "unknown option raises an error" (string? bad-option))

(define bad-max
  (xpcall (lambda () (llm-chat-generate ctx smpl msgs (lambda (piece) #t)
                                        '((max-tokens . -1))))
          (lambda (e) e)))
(check-true "negative max-tokens raises an error" (string? bad-max))

(define keep-kv
  (xpcall (lambda () (llm-chat-generate ctx smpl msgs (lambda (piece) #t)
                                        '((keep-kv . #t))))
          (lambda (e) e)))
(check-true "keep-kv raises an error" (string? keep-kv))

;; ---- lifetime ------------------------------------------------------------

(check "sampler free is idempotent" (llm-sampler-free smpl) #t)
(check "context free is idempotent" (llm-context-free ctx) #t)

(define uaf
  (xpcall (lambda () (llm-chat-generate ctx smpl msgs (lambda (piece) #t)))
          (lambda (e) e)))
(check-true "use-after-free raises an error" (string? uaf))

(check "tiny sampler free is idempotent" (llm-sampler-free tsmpl) #t)
(check "tiny context free is idempotent" (llm-context-free tctx) #t)
(check "tiny model free is idempotent" (llm-model-free tiny) #t)
(check "model free is idempotent" (llm-model-free m) #t)

(display "all chat generation tests passed") (newline)
