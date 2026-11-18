;; Chat template binding test.
;;
;;   ./build/scmllama test/chat.scm

(require "test.common")

(define m (llm-model-load model-path))

;; ---- built-in template list ---------------------------------------------

(define names (llm-chat-templates))

(display "templates  : ") (write (vector-length names)) (newline)
(display "templates is a vector") (display ": ") (write (vector? names)) (newline)
(display "templates is non-empty") (display ": ") (write (> (vector-length names) 0)) (newline)
(display "templates lists chatml") (display ": ") (write (member "chatml" (vector->list names))) (newline)

;; ---- default template from model metadata -------------------------------
;;
;; Qwen3 ships a ChatML-style template, which llama.cpp renders as:
;;   <|im_start|>role\ncontent<|im_end|>\n
;; with an assistant header appended when add-assistant is true.

(define msgs '(("system" . "You are a helpful assistant.")
               ("user" . "Hello!")))

(define prompt (llm-chat-template-apply m msgs))

(display "prompt     : ") (write prompt) (newline)

(display "chatml render") (display ": ") (write prompt) (newline)

(display "add-assistant #f stops after the user turn") (display ": ") (write (llm-chat-template-apply m msgs '((add-assistant . #f)))) (newline)

;; ---- explicit template name ---------------------------------------------

(display "explicit built-in name") (display ": ") (write (llm-chat-template-apply m '(("user" . "Hi"))
                                '((template . "chatml")))) (newline)

;; ---- empty message list --------------------------------------------------

(display "empty message list") (display ": ") (write (llm-chat-template-apply m '())) (newline)

;; ---- error paths ---------------------------------------------------------

;; Wrong message shape: a two-element list is not a (role . content) pair.
(define bad-msg
  (xpcall (lambda () (llm-chat-template-apply m '(("user" "Hi"))))
          (lambda (e) e)))
(display "malformed message raises an error") (display ": ") (write bad-msg) (newline)

(define bad-role
  (xpcall (lambda () (llm-chat-template-apply m '((user . "Hi"))))
          (lambda (e) e)))
(display "non-string role raises an error") (display ": ") (write bad-role) (newline)

(define bad-list
  (xpcall (lambda () (llm-chat-template-apply m "not a list"))
          (lambda (e) e)))
(display "non-list messages raises an error") (display ": ") (write bad-list) (newline)

(define bad-option
  (xpcall (lambda () (llm-chat-template-apply m msgs '((nope . 1))))
          (lambda (e) e)))
(display "unknown option raises an error") (display ": ") (write bad-option) (newline)

(define bad-template
  (xpcall (lambda () (llm-chat-template-apply m msgs
                                              '((template . 42))))
          (lambda (e) e)))
(display "non-string template raises an error") (display ": ") (write bad-template) (newline)

(define unsupported
  (xpcall (lambda () (llm-chat-template-apply m msgs
                                              '((template . "not-a-template"))))
          (lambda (e) e)))
(display "unsupported template raises an error") (display ": ") (write unsupported) (newline)

;; ---- lifetime ------------------------------------------------------------

(display "free is idempotent") (display ": ") (write (llm-model-free m)) (newline)

(define uaf
  (xpcall (lambda () (llm-chat-template-apply m msgs)) (lambda (e) e)))
(display "use-after-free raises an error") (display ": ") (write uaf) (newline)

(display "done") (newline)
