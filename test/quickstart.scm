;; The shortest useful example: load a model, ask one question, stream the
;; answer to stdout. Run it with ./build/scmllama test/quickstart.scm.

(define m    (llm-model-load "models/Qwen3-0.6B-Q4_K_M.gguf"))
(define ctx  (llm-context-new m '((n-ctx . 1024))))
(define smpl (llm-sampler-new m '((temp . 0.7))))

(llm-chat-generate ctx smpl
  '(("system" . "You are a village blacksmith. Answer in one short sentence.")
    ("user"   . "Do you have a sword for me? /no_think"))
  (lambda (piece) (display piece) #t))

(newline)
(llm-sampler-free smpl)
(llm-context-free ctx)
(llm-model-free m)
