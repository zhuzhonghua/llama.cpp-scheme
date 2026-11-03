cmake_minimum_required(VERSION 3.14...3.28)

project(scmllama LANGUAGES C CXX)

# 和 llama.cpp 保持一致：C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 不关闭 GNU 扩展：ggml 的 CPU/Metal 后端会用到（llama 自己也是 gnu++17）

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# llama.cpp（submodule）：只编库，不编工具/示例/server
# ---------------------------------------------------------------------------
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# dummyscheme（submodule）：直接编源码，不含它自己的 main.cpp
# ---------------------------------------------------------------------------
set(DUMMYSCHEME_SOURCES
    dummyscheme/vm.cpp
    dummyscheme/scmapi.cpp
    dummyscheme/scmbasic.cpp
    dummyscheme/scmmath.cpp
    dummyscheme/scmstring.cpp
    dummyscheme/scmcompiler.cpp
    dummyscheme/scmport.cpp
    dummyscheme/scmvector.cpp
    dummyscheme/scmtable.cpp
    dummyscheme/scminit.cpp
    dummyscheme/scmdebug.cpp)

add_executable(scmllama
    main.cpp
    scmllama.cpp
    ${DUMMYSCHEME_SOURCES})

target_include_directories(scmllama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dummyscheme)

target_compile_features(scmllama PRIVATE cxx_std_17)

target_link_libraries(scmllama PRIVATE
    llama
    Threads::Threads)
