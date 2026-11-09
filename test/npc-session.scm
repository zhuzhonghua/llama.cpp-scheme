;; npc-session.scm — Scheme 侧会话层：model 之外的所有 llm 句柄都藏在闭包里。
;;
;;   (require "test.npc-session")
;;   (define s (make-session model
;;                           "你是一个站在城门口的卫兵……"
;;                           '((n-ctx . 1024))              ; context 选项
;;                           '((temp . 0.8) (top-k . 40))   ; sampler 选项
;;                           '((max-tokens . 200))))        ; 生成选项（可省略）
;;   (s 'say "你好")   ;=> 台词字符串（多轮追加，不重算历史）
;;   (s 'kv-pos)       ;=> 当前 KV 占用的 token 数
;;   (s 'history)      ;=> ((玩家 . 你好) (卫兵 . "…")) 旧→新
;;   (s 'reset)        ;=> #t  清空 KV 与历史，回到首次对话状态
;;   (s 'free)         ;=> #t  释放 sampler 与 context
;;
;; 为什么放在 Scheme 而不是 C++：
;;   C++ 侧保持与 llama.cpp 的薄绑定（model/context/sampler/batch 全部显式
;;   句柄）；会话状态、对话历史、上下文溢出策略都属于业务逻辑，因此放在
;;   Scheme。调用方只拿到一个过程，既拿不到 ctx 也不会传错句柄。
;;
;; 多轮机制：第一轮把 system 块连同首个 user 块一起 prefill；之后每轮只追加
;;   [<|im_end|>] <|im_start|>user … <|im_end|> <|im_start|>assistant <think></think>
;; 并带 (keep-kv . #t)，历史 token 不会重算。
;;
;; 溢出策略：不做前缀淘汰，而是 reset 后只保留最近几轮重建 prompt。
;;   原因有二：system 块被钉在 position 0，前缀淘汰会把它一并删掉；且
;;   llm-kv-shift 只挪 position、不重算已缓存的 K，长程属近似。重建是精确的，
;;   代价只是每溢出 N 轮重算一次历史（比每轮都重算便宜得多）。
;;   需要更细粒度的控制时，llm-kv-remove / llm-kv-shift 仍可直接调用。

;; 读取 alist 选项，缺省时返回 dflt。
(define (npc-session-opt opts key dflt)
  (let ((hit (assq key opts)))
    (if hit (cdr hit) dflt)))

;; 取列表前 n 个元素（列表可能更短）。
(define (npc-session-take lst n)
  (if (or (<= n 0) (null? lst))
      '()
      (cons (car lst) (npc-session-take (cdr lst) (- n 1)))))

;; 去掉 <think>…</think> 块（Qwen3 在非推理模式下仍可能吐出来）。
(define (npc-session-strip-think text)
  (let ((start (npc-session-index text "<think>")))
    (if (not start)
        text
        (let ((end (npc-session-index-from text "</think>" (+ start 7))))
          (if (not end)
              (substring text 0 start)
              (string-append
                (substring text 0 start)
                (npc-session-strip-think
                  (substring text (+ end 8) (string-length text)))))))))

(define (npc-session-index haystack needle)
  (npc-session-index-from haystack needle 0))

(define (npc-session-index-from haystack needle from)
  (let ((lh (string-length haystack)) (ln (string-length needle)))
    (let loop ((i from))
      (cond ((> (+ i ln) lh) #f)
            ((string=? (substring haystack i (+ i ln)) needle) i)
            (else (loop (+ i 1)))))))

(define (npc-session-space? c)
  (or (char=? c #\space) (char=? c #\newline)
      (char=? c #\tab) (char=? c #\return)))

(define (npc-session-trim s)
  (let* ((n (string-length s))
         (start (let loop ((i 0))
                  (cond ((= i n) n)
                        ((npc-session-space? (string-ref s i)) (loop (+ i 1)))
                        (else i))))
         (end (let loop ((i (- n 1)))
                (cond ((< i 0) 0)
                      ((npc-session-space? (string-ref s i)) (loop (- i 1)))
                      (else (+ i 1))))))
    (if (>= start end) "" (substring s start end))))

;; make-session : model × string × alist × alist × [alist] → procedure
(define (make-session model system-prompt ctx-opts sampler-opts . rest)
  (let ((gen-opts (if (null? rest) '() (car rest)))
        (ctx (llm-context-new model ctx-opts))
        (smpl (llm-sampler-new model sampler-opts)))
    (let ((n-ctx (npc-session-opt ctx-opts 'n-ctx 1024))
          (max-tokens (npc-session-opt gen-opts 'max-tokens 200))
          (keep-pairs 4)
          (pairs '())      ; ((user . reply) …) 新→旧
          (unborn? #t)     ; 还没 prefill 过 system 块
          (need-close? #f)) ; 上一轮回复尚未写 <|im_end|>

      (define (kv-pos) (llm-kv-pos ctx))

      (define (system-block)
        (string-append "<|im_start|>system\n" system-prompt "<|im_end|>\n"))

      (define (user-block text)
        (string-append "<|im_start|>user\n" text "<|im_end|>\n"))

      (define (assistant-block text)
        (string-append "<|im_start|>assistant\n" text "<|im_end|>\n"))

      ;; Qwen3 关闭思考：预填空 <think> 块，模型会直接给出台词。
      (define (assistant-header)
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")

      (define (estimate text)
        (length (llm-tokenize model text
                              '((add-special . #f) (parse-special . #t)))))

      ;; 本轮要追加的增量。首轮带上 system 块。
      (define (turn-chunk text)
        (string-append
          (if unborn? (system-block) (if need-close? "<|im_end|>\n" ""))
          (user-block text)
          (assistant-header)))

      (define (turn-gen-opts)
        (append (list (cons 'add-special (if unborn? #t #f))
                      (cons 'parse-special #t)
                      (cons 'keep-kv #t))
                gen-opts))

      ;; 重建后的增量：上一轮回复已由重建补上 <|im_end|>，因此不再需要闭合。
      (define (fresh-chunk text)
        (string-append (user-block text) (assistant-header)))

      (define (join strings)
        (if (null? strings) "" (string-append (car strings) (join (cdr strings)))))

      ;; 重建：reset 后用 system + 最近 k 轮文本重新 prefill，不生成。
      (define (rebuild! k)
        (llm-reset ctx)
        (let ((seed (join (cons (system-block)
                                (map (lambda (p)
                                       (string-append
                                         (user-block (car p))
                                         (if (cdr p) (assistant-block (cdr p)) "")))
                                     (reverse (npc-session-take pairs k)))))))
          (let ((batch (llm-batch-from-tokens
                         (llm-tokenize model seed
                                       '((add-special . #t) (parse-special . #t)))
                         '((last-logits . #f)))))
            (llm-decode ctx batch)
            (llm-batch-free batch)))
        (set! pairs (npc-session-take pairs k))
        (set! unborn? #f)
        (set! need-close? #f))

      ;; 保证 seed + 本轮增量 + 回复空间能装进 n-ctx，只在装不下时才重建。
      (define (ensure-room! text)
        (if (or unborn?
                (<= (+ (kv-pos) (estimate (turn-chunk text)) max-tokens) n-ctx))
            #t
            (let loop ((k (let ((n (length pairs)))
                            (if (< n keep-pairs) n keep-pairs))))
              (rebuild! k)
              (if (or (= k 0)
                      (<= (+ (kv-pos) (estimate (fresh-chunk text)) max-tokens)
                          n-ctx))
                  #t
                  (loop (- k 1))))))

      (define (say text)
        (ensure-room! text)
        (let* ((chunk (turn-chunk text))
               (raw (llm-generate ctx smpl chunk
                                  (lambda (piece) #t)
                                  (turn-gen-opts)))
               (reply (npc-session-trim (npc-session-strip-think raw))))
          (set! unborn? #f)
          (set! need-close? #t)
          (set! pairs (cons (cons text reply) pairs))
          reply))

      (define (do-reset)
        (llm-reset ctx)
        (set! pairs '())
        (set! unborn? #t)
        (set! need-close? #f)
        #t)

      (lambda (msg . args)
        (case msg
          ((say)     (say (car args)))
          ((kv-pos)  (kv-pos))
          ((history) (reverse pairs))
          ((reset)   (do-reset))
          ((free)    (llm-sampler-free smpl)
                     (llm-context-free ctx)
                     #t)
          (else (error "npc-session: unknown message" msg)))))))
