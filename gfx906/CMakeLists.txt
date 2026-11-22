add_executable(gfx906-test-tp test-gfx906-tp.cpp)
target_link_libraries(gfx906-test-tp PRIVATE ggml)

add_executable(gfx906-test-flashnext-moe test-flashnext-moe.cpp)
target_link_libraries(gfx906-test-flashnext-moe PRIVATE ggml)
