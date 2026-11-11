;; Chat template binding test.
;;
;;   ./build/scmllama test/chat.scm

(require "test.common")

(define m (llm-model-load model-path))

;; ---- built-in template list ---------------------------------------------

(define names (llm-chat-templates))

(display "templates  : ") (write (vector-length names)) (newline)
(check-true "templates is a vector" (vector? names))
(check-true "templates is non-empty" (> (vector-length names) 0))
(check-true "templates lists chatml" (member "chatml" (vector->list names)))

;; ---- default template from model metadata -------------------------------
;;
;; Qwen3 ships a ChatML-style template, which llama.cpp renders as:
;;   <|im_start|>role\ncontent<|im_end|>\n
;; with an assistant header appended when add-assistant is true.

(define msgs '(("system" . "You are a helpful assistant.")
               ("user" . "Hello!")))

(define prompt (llm-chat-template-apply m msgs))

(display "prompt     : ") (write prompt) (newline)

(check "chatml render"
       prompt
       (string-append
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello!<|im_end|>\n"
        "<|im_start|>assistant\n"))

(check "add-assistant #f stops after the user turn"
       (llm-chat-template-apply m msgs '((add-assistant . #f)))
       (string-append
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello!<|im_end|>\n"))

;; ---- explicit template name ---------------------------------------------

(check "explicit built-in name"
       (llm-chat-template-apply m '(("user" . "Hi"))
                                '((template . "chatml")))
       "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")

;; ---- empty message list --------------------------------------------------

(check "empty message list"
       (llm-chat-template-apply m '())
       "<|im_start|>assistant\n")

;; ---- error paths ---------------------------------------------------------

;; Wrong message shape: a two-element list is not a (role . content) pair.
(define bad-msg
  (xpcall (lambda () (llm-chat-template-apply m '(("user" "Hi"))))
          (lambda (e) e)))
(check-true "malformed message raises an error" (string? bad-msg))

(define bad-role
  (xpcall (lambda () (llm-chat-template-apply m '((user . "Hi"))))
          (lambda (e) e)))
(check-true "non-string role raises an error" (string? bad-role))

(define bad-list
  (xpcall (lambda () (llm-chat-template-apply m "not a list"))
          (lambda (e) e)))
(check-true "non-list messages raises an error" (string? bad-list))

(define bad-option
  (xpcall (lambda () (llm-chat-template-apply m msgs '((nope . 1))))
          (lambda (e) e)))
(check-true "unknown option raises an error" (string? bad-option))

(define bad-template
  (xpcall (lambda () (llm-chat-template-apply m msgs
                                              '((template . 42))))
          (lambda (e) e)))
(check-true "non-string template raises an error" (string? bad-template))

(define unsupported
  (xpcall (lambda () (llm-chat-template-apply m msgs
                                              '((template . "not-a-template"))))
          (lambda (e) e)))
(check-true "unsupported template raises an error" (string? unsupported))

;; ---- lifetime ------------------------------------------------------------

(check "free is idempotent" (llm-model-free m) #t)

(define uaf
  (xpcall (lambda () (llm-chat-template-apply m msgs)) (lambda (e) e)))
(check-true "use-after-free raises an error" (string? uaf))

(display "all chat template tests passed") (newline)
