;; Session layer tests: ctx/sampler hidden inside a closure.

(require "test.common")
(require "test.npc-session")

(define model (llm-model-load model-path '((n-gpu-layers . 99))))

(define session
  (make-session model
                "你是一个城门口的卫兵，只回答一句话。"
                '((n-ctx . 256))
                '((greedy . #t))
                '((max-tokens . 24))))

(display "kv starts empty") (display ": ") (write (session 'kv-pos)) (newline)
(display "history starts empty") (display ": ") (write (length (session 'history))) (newline)

(define reply-1 (session 'say "你好"))
(display "first reply is non-empty") (display ": ") (write (> (string-length reply-1) 0)) (newline)
(define pos-1 (session 'kv-pos))
(display "kv grew after turn 1") (display ": ") (write (> pos-1 0)) (newline)

(define reply-2 (session 'say "你叫什么名字"))
(display "second reply is non-empty") (display ": ") (write (> (string-length reply-2) 0)) (newline)
(define pos-2 (session 'kv-pos))
(display "kv appended, not re-prefilled") (display ": ") (write (> pos-2 pos-1)) (newline)
(display "history has two turns") (display ": ") (write (length (session 'history))) (newline)

;; 多跑几轮把 256 的上下文顶爆，应当自动重建而不是报错。
(define (run turns)
  (if (> turns 0)
      (begin (session 'say "继续") (run (- turns 1)))
      #t))
(run 40)
(display "kv stays under n-ctx") (display ": ") (write (< (session 'kv-pos) 256)) (newline)
(display "history kept after rebuild") (display ": ") (write (> (length (session 'history)) 0)) (newline)
;; 一共说了 42 轮却远少于 42 条，说明旧轮次被淘汰了。
(display "old turns evicted") (display ": ") (write (< (length (session 'history)) 20)) (newline)

(session 'reset)
(display "kv reset to zero") (display ": ") (write (session 'kv-pos)) (newline)
(display "history cleared") (display ": ") (write (length (session 'history))) (newline)

(display "free returns true") (display ": ") (write (session 'free)) (newline)
(llm-model-free model)

(display "npc-session tests passed\n")
