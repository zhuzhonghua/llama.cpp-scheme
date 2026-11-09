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

(check "kv starts empty" (session 'kv-pos) 0)
(check "history starts empty" (length (session 'history)) 0)

(define reply-1 (session 'say "你好"))
(check-true "first reply is non-empty" (> (string-length reply-1) 0))
(define pos-1 (session 'kv-pos))
(check-true "kv grew after turn 1" (> pos-1 0))

(define reply-2 (session 'say "你叫什么名字"))
(check-true "second reply is non-empty" (> (string-length reply-2) 0))
(define pos-2 (session 'kv-pos))
(check-true "kv appended, not re-prefilled" (> pos-2 pos-1))
(check "history has two turns" (length (session 'history)) 2)

;; 多跑几轮把 256 的上下文顶爆，应当自动重建而不是报错。
(define (run turns)
  (if (> turns 0)
      (begin (session 'say "继续") (run (- turns 1)))
      #t))
(run 40)
(check-true "kv stays under n-ctx" (< (session 'kv-pos) 256))
(check-true "history kept after rebuild" (> (length (session 'history)) 0))
;; 一共说了 42 轮却远少于 42 条，说明旧轮次被淘汰了。
(check-true "old turns evicted" (< (length (session 'history)) 20))

(session 'reset)
(check "kv reset to zero" (session 'kv-pos) 0)
(check "history cleared" (length (session 'history)) 0)

(check "free returns true" (session 'free) #t)
(llm-model-free model)

(display "npc-session tests passed\n")
