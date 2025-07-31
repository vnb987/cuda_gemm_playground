cmake_minimum_required(VERSION 3.18)
include(FetchContent)

set(THIRDPARTY_RMM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rmm)

FetchContent_Declare(
  rmm
  GIT_REPOSITORY https://github.com/rapidsai/rmm.git
  GIT_TAG        branch-24.06
  SOURCE_DIR     ${THIRDPARTY_RMM_DIR}
)

FetchContent_MakeAvailable(rmm)
