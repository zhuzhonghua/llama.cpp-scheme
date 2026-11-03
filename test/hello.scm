;; 最小冒烟测试：证明 Scheme VM 已经链接上 libllama 并能调用 C API。
;;
;;   ./build/scmllama test/hello.scm
;;
;; 后端由宿主侧的 scmllama::Runtime 用 RAII 管理，Scheme 不需要手动 init/free。

(display "llama.cpp version : ")
(display (llm-version))
(newline)

(display "system info:")
(newline)
(display (llm-system-info))
(newline)

(display "hello, llama.cpp from Scheme!")
(newline)
