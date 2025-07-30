# unittest/CMakeLists.txt
include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.8.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Unit test source
add_executable(gemm_test test_gemm.cpp)

# Link with GTest and the gemm library
target_link_libraries(gemm_test
    PRIVATE
        gemm
        gtest
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(gemm_test)
