;; npc-demo.scm — 1 NPC + 1 quest state machine.
;;
;;   ./build/scmllama test/npc-demo.scm
;;
;; Why this shape demonstrates the point of embedding an LLM in a Scheme
;; host, rather than calling a hosted LLM API from an app:
;;
;;   1. LLM proposes   — a first generation classifies the player's intent
;;                       (ACCEPT / GIVE / CHAT).
;;   2. Scheme judges  — apply-intent validates the proposal against the
;;                       quest state and either applies it or rejects it.
;;   3. LLM narrates   — a second generation writes the NPC line, informed
;;                       by the validated result (pass / reject + reason).
;;
;; Everything runs in one process, offline: no server, no Python, no IPC.
;; The model can propose, but it can never change state — only Scheme does,
;; and only when the transition is legal.
;;
;; Commands (free text also works — it is classified as an intent):
;;   <any text>      talk to the NPC in natural language
;;   talk <text>     same, kept for habit
;;   accept          force the ACCEPT intent (bypasses the classifier)
;;   give iron-ore   force the GIVE intent (bypasses the classifier)
;;   inventory       show what you are carrying
;;   quest           show the current quest state
;;   help            list commands
;;   quit            exit

(require "test.common")

;; ---- model / context / samplers ------------------------------------------

(define m   (llm-model-load model-path))
(define ctx (llm-context-new m '((n-ctx . 1024))))

;; Dialogue sampler: a little warmth, but not so much that a 0.6B model
;; wanders off. No fixed seed on purpose — each session differs.
(define smpl (llm-sampler-new m '((temp . 0.85) (top-k . 40) (top-p . 0.92)
                                   (penalty-last-n . 64)
                                   (penalty-repeat . 1.15)
                                   (penalty-freq . 0.4)
                                   (penalty-present . 0.2))))

;; Intent classifier: greedy, we want the single most likely tag.
(define smpl-intent (llm-sampler-new m '((greedy . #t))))

;; ---- NPC definition -------------------------------------------------------

(define npc-name "老王")

(define persona
  "你正在扮演老王，村子东头的铁匠。老王性格豪爽、说话直来直去，
常用『哼』『嘿』『得嘞』这类口头语。他正打算给村里的守卫打一把剑，
但手头缺铁矿石，正为此发愁。")

(define system-instruction
  "你就是老王。只输出老王对玩家说的一句话台词，不要加引号、不要加『老王：』前缀，
不要使用 <think> 标签、不要思考、不要分析。不要替玩家说话。")

;; ---- quest state (deterministic, Scheme-owned) ----------------------------

(define quest-state 'not-offered)   ; not-offered -> accepted -> completed
(define iron-ore 0)                  ; how many iron ore handed to the smith
(define carried 0)                   ; ore the player carries, not yet handed in

(define (quest-state-text)
  (case quest-state
    ((not-offered) "任务尚未接受")
    ((accepted)    (string-append
                    "已接受任务：收集 3 块铁矿石（已交 "
                    (number->string iron-ore) " / 3）"))
    ((completed)   "任务已完成，老王已赠予你一把铁剑")
    (else "未知")))

;; ---- intent: the LLM proposes, Scheme judges ------------------------------

(define intent-system
  "你是意图分类器。只输出一个标签：ACCEPT、GIVE 或 CHAT，不要输出任何其它内容。
ACCEPT：玩家接受任务（好、我接了、没问题、行、成交）
GIVE：玩家把物品交给老王（给你、拿去、这块矿石给你）
CHAT：其它（打招呼、提问、闲聊、拒绝）

示例：
玩家：我接了
ACCEPT
玩家：给你一块矿石
GIVE
玩家：今天天气不错
CHAT")

(define (classify-intent text)
  (llm-reset ctx)
  (let ((raw (llm-generate ctx smpl-intent
                           (string-append
                             "<|im_start|>system\n"
                             intent-system "<|im_end|>\n"
                             "<|im_start|>user\n玩家：" text "<|im_end|>\n"
                             "<|im_start|>assistant\n<think>\n\n</think>\n\n")
                           (lambda (piece) #t)
                           '((max-tokens . 8) (parse-special . #t)))))
    (cond ((string-index raw "ACCEPT") 'accept)
          ((string-index raw "GIVE") 'give)
          ((string-index raw "CHAT") 'chat)
          ;; Fallback if the model ignored the protocol entirely.
          ((or (string-index raw "接受") (string-index raw "接了")) 'accept)
          ((or (string-index raw "给") (string-index raw "拿去")) 'give)
          (else 'chat))))

;; Apply a proposed intent. Returns (status . reason) where status is one of
;; accepted / rejected / ignored. This is the only place that mutates state.
(define (apply-intent intent)
  (cond ((eq? intent 'accept)
         (case quest-state
           ((not-offered)
            (begin
              (set! quest-state 'accepted)
              (cons 'accepted "玩家接受了任务")))
           ((accepted)  (cons 'ignored "任务已经接过了"))
           ((completed) (cons 'rejected "任务早已完成"))
           (else        (cons 'ignored "未知状态"))))
        ((eq? intent 'give)
         (case quest-state
           ((accepted)
            (cond ((<= carried 0)
                   (cons 'rejected "玩家背上没有矿石"))
                  (else
                   (begin
                     (set! carried (- carried 1))
                     (set! iron-ore (+ iron-ore 1))
                     (if (>= iron-ore 3)
                         (begin
                           (set! quest-state 'completed)
                           (cons 'accepted "收下第 3 块铁矿石，任务完成，交给玩家一把铁剑"))
                         (cons 'accepted
                               (string-append "收下 1 块铁矿石（"
                                              (number->string iron-ore) " / 3）")))))))
           ((not-offered) (cons 'rejected "玩家还没接任务，老王不能收矿石"))
           ((completed)   (cons 'rejected "任务已完成，不再收矿石"))
           (else          (cons 'ignored "未知状态"))))
        (else (cons 'ignored "闲聊，不影响状态"))))

(define (intent-tag intent)
  (case intent
    ((accept) "ACCEPT")
    ((give)   "GIVE")
    (else     "CHAT")))

(define (status-text status)
  (case status
    ((accepted) "校验通过")
    ((rejected) "校验拒绝")
    (else       "无需处理")))

;; Visible proof that the model proposes and Scheme decides.
(define (report-verdict intent verdict)
  (display "  · LLM 判定意图 ")
  (display (intent-tag intent))
  (display " → Scheme ")
  (display (status-text (car verdict)))
  (display "（")
  (display (cdr verdict))
  (display "）\n")
  (flush-output))

;; ---- dialogue history -----------------------------------------------------

(define history '())   ; ((speaker . text) ...) most-recent-first

(define (take lst n)
  (cond ((or (<= n 0) (null? lst)) '())
        (else (cons (car lst) (take (cdr lst) (- n 1))))))

(define (history-add speaker text)
  (set! history (cons (cons speaker text) history))
  (when (> (length history) 12)
    (set! history (take history 12))))

(define (history-text)
  (apply string-append
         (map (lambda (pair)
                (string-append (car pair) "：" (cdr pair) "\n"))
              (reverse history))))

;; ---- prompt builder -------------------------------------------------------

(define (verdict-hint status)
  (case status
    ((accepted) "老王的请求得到了满足，简短回应，语气高兴。")
    ((rejected) "老王必须拒绝，例如：不行，你还没接这活儿呢。")
    (else       "只是闲聊，老王正常回应即可。")))

(define (build-prompt player-input intent verdict)
  (string-append
    "<|im_start|>system\n"
    persona "\n"
    system-instruction "<|im_end|>\n"
    "<|im_start|>user\n"
    "当前任务状态：" (quest-state-text) "\n"
    "玩家背包：铁矿石 x" (number->string carried) "\n"
    "刚刚的判定：玩家意图 " (intent-tag intent)
    "，Scheme 结果=" (status-text (car verdict))
    "，原因=" (cdr verdict) "\n"
    (verdict-hint (car verdict)) "\n\n"
    "对话记录：\n"
    (history-text)
    "玩家：" player-input "<|im_end|>\n"
    ;; Qwen3 "thinking off" chat template (enable_thinking=false): pre-fill
    ;; an empty <think></think> block so the model answers directly instead
    ;; of opening its own block, which derails a 0.6B model into garbage.
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"))

;; Strip <think>...</think> blocks from the model output.
(define (strip-think text)
  (let ((start (string-index text "<think>")))
    (if (not start)
        text
        (let ((end (string-index-from text "</think>" (+ start 7))))
          (if (not end)
              (substring text 0 start)
              (string-append (substring text 0 start)
                             (strip-think (substring text (+ end 8) (string-length text)))))))))

;; Find the first occurrence of `needle` in `haystack`, return its index or #f.
(define (string-index haystack needle)
  (let ((lh (string-length haystack)) (ln (string-length needle)))
    (let loop ((i 0))
      (cond ((> (+ i ln) lh) #f)
            ((string=? (substring haystack i (+ i ln)) needle) i)
            (else (loop (+ i 1)))))))

(define (string-index-from haystack needle from)
  (let ((lh (string-length haystack)) (ln (string-length needle)))
    (let loop ((i from))
      (cond ((> (+ i ln) lh) #f)
            ((string=? (substring haystack i (+ i ln)) needle) i)
            (else (loop (+ i 1)))))))

(define (whitespace? c)
  (or (char=? c #\space) (char=? c #\newline) (char=? c #\tab)
      (char=? c #\return)))

(define (string-prefix? s prefix)
  (let ((ls (string-length s)) (lp (string-length prefix)))
    (and (>= ls lp) (string=? (substring s 0 lp) prefix))))

;; Scan the first `limit` bytes for a "：" / ":" label separator.
;; Returns the offset just past the colon, or #f. Substring comparison is
;; used instead of #\： literals: dummyscheme's reader rejects full-width
;; char literals.
(define (find-speaker-colon-end s limit)
  (let ((n (min limit (string-length s)))
        (len (string-length s)))
    (let loop ((i 0))
      (cond ((>= i n) #f)
            ((and (<= (+ i 3) len)
                  (string=? (substring s i (+ i 3)) "："))
             (+ i 3))
            ((string=? (substring s i (+ i 1)) ":") (+ i 1))
            (else (loop (+ i 1)))))))

;; Drop a leading speaker label the model sometimes echoes back, e.g.
;; "老王：", "王：", or a garbled variant. Any colon within the first
;; 12 characters is treated as the label separator.
(define (strip-speaker text)
  (let loop ((s text))
    (cond ((string-prefix? s "<|im_start|>assistant")
           (loop (substring s 22 (string-length s))))
          (else
           (let ((end (find-speaker-colon-end s 12)))
             (if end
                 (string-trim (substring s end (string-length s)))
                 s))))))

(define (string-trim s)
  (let ((n (string-length s)))
    (let ((start (let loop ((i 0))
                   (cond ((= i n) n)
                         ((whitespace? (string-ref s i)) (loop (+ i 1)))
                         (else i))))
          (end (let loop ((i (- n 1)))
                 (cond ((< i 0) 0)
                       ((whitespace? (string-ref s i)) (loop (- i 1)))
                       (else (+ i 1))))))
      (if (>= start end) "" (substring s start end))))) 

;; ---- one conversational turn ---------------------------------------------

(define (npc-reply player-input intent verdict)
  (llm-reset ctx)
  ;; Print the speaker prefix immediately so the user sees the game is
  ;; working while the model generates (generation is silent otherwise).
  (display npc-name) (display "：") (flush-output)
  (let ((raw (llm-generate ctx smpl (build-prompt player-input intent verdict)
                           (lambda (piece) #t)
                           '((max-tokens . 200) (parse-special . #t)))))
    (let ((text (strip-speaker (string-trim (strip-think raw)))))
      (display text) (newline)
      (history-add "玩家" player-input)
      (history-add npc-name text)
      text)))

;; Classify, judge, then narrate.
(define (converse player-input)
  (let ((intent (classify-intent player-input)))
    (let ((verdict (apply-intent intent)))
      (report-verdict intent verdict)
      (npc-reply player-input intent verdict))))

;; For the explicit accept / give commands: skip the classifier.
(define (converse-forced player-input intent)
  (let ((verdict (apply-intent intent)))
    (report-verdict intent verdict)
    (npc-reply player-input intent verdict)))

;; ---- command parsing ------------------------------------------------------

;; Read one line from stdin. Returns #f on EOF, "" for an empty line.
;; Do NOT use peek-char to detect EOF: in interactive mode it blocks
;; waiting for the next character, which deadlocks the input loop.
(define (read-line)
  (let loop ((c (read-char)) (acc '()))
    (cond ((eof-object? c)
           (if (null? acc) #f (list->string (reverse acc))))
          ((char=? c #\newline)
           (list->string (reverse acc)))
          (else
           (loop (read-char) (cons c acc))))))

(define (handle-command line)
  (cond ((string=? line "") #t)

        ((or (string=? line "quit") (string=? line "exit"))
         (display "再见。\n") #f)

        ((string=? line "help")
         (display "可用命令：\n")
         (display "  <任意文字>      直接和老王说话（LLM 判意图，Scheme 校验）\n")
         (display "  talk <文字>     同上\n")
         (display "  accept          强制 ACCEPT 意图（跳过分类器，便于测试）\n")
         (display "  give iron-ore   强制 GIVE 意图（跳过分类器，便于测试）\n")
         (display "  pick iron-ore   在河边滩上捡一块铁矿石\n")
         (display "  inventory       查看背包\n")
         (display "  quest           查看任务状态\n")
         (display "  help            显示帮助\n")
         (display "  quit            退出\n") #t)

        ((string=? line "inventory")
         (display "背包：铁矿石 x")
         (display (number->string carried))
         (display "（已交给老王 ")
         (display (number->string iron-ore))
         (display " 块）\n") #t)

        ((string=? line "quest")
         (display "任务：") (display (quest-state-text)) (newline) #t)

        ((string=? line "accept")
         (converse-forced "我接受你的请求" 'accept) #t)

        ((string=? line "give iron-ore")
         (converse-forced "给你铁矿石" 'give) #t)

        ((string=? line "pick iron-ore")
         (set! carried (+ carried 1))
         (display "（你在河边滩上捡到一块铁矿石）\n") #t)

        ((or (string=? line "talk") (string-prefix? line "talk "))
         (converse (if (string=? line "talk")
                       "你好"
                       (substring line 5 (string-length line)))) #t)

        ;; Anything else is natural language for the NPC.
        (else (converse line) #t)))

;; ---- main loop ------------------------------------------------------------

(display "=== 铁匠老王的小铺 ===\n")
(display "你走进村子东头的铁匠铺，炉火正旺。\n")
(display "直接说话即可，比如「我接了」「给你一块矿石」；想要矿石先「pick iron-ore」。\n")
(display "LLM 只负责提议意图和写台词，状态由 Scheme 校验后变更。\n\n")

(define (main-loop)
  (display "> ") (flush-output)
  (let ((line (read-line)))
    (cond ((not line)
           (display "\n再见。\n")
           #f)
          ((handle-command line) (main-loop))
          (else #t))))

(main-loop)

;; ---- cleanup --------------------------------------------------------------

(llm-sampler-free smpl-intent)
(llm-sampler-free smpl)
(llm-context-free ctx)
(llm-model-free m)
