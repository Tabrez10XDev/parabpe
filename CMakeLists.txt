cmake_minimum_required(VERSION 3.20)
project(parabpe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PARABPE_NATIVE "Tune for the host CPU" ON)
option(PARABPE_ASAN "Build with AddressSanitizer + UBSan" OFF)
option(PARABPE_TSAN "Build with ThreadSanitizer (use clang + LLVM libomp)" OFF)

# ---- OpenMP ----------------------------------------------------------------
# Apple Clang has no bundled OpenMP; point FindOpenMP at Homebrew's libomp.
if(APPLE AND NOT OpenMP_CXX_FLAGS)
  find_program(BREW brew)
  if(BREW)
    execute_process(COMMAND ${BREW} --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
  endif()
endif()
find_package(OpenMP REQUIRED)

# ---- PCRE2 (8-bit) -------------------------------------------------------------
find_path(PCRE2_INCLUDE_DIR pcre2.h HINTS /opt/homebrew/opt/pcre2/include /usr/local/opt/pcre2/include)
find_library(PCRE2_LIBRARY pcre2-8 HINTS /opt/homebrew/opt/pcre2/lib /usr/local/opt/pcre2/lib)
if(NOT PCRE2_INCLUDE_DIR OR NOT PCRE2_LIBRARY)
  message(FATAL_ERROR "PCRE2 not found (brew install pcre2 / apt install libpcre2-dev)")
endif()
# SPEC.md §2 pins Unicode 16.0, which needs PCRE2 >= 10.45 (the runtime check is in test_smoke.cpp).
file(STRINGS "${PCRE2_INCLUDE_DIR}/pcre2.h" PCRE2_VERSION_LINES REGEX "^#define PCRE2_(MAJOR|MINOR)[ \t]")
string(REGEX REPLACE ".*PCRE2_MAJOR[ \t]+([0-9]+).*" "\\1" PCRE2_MAJOR "${PCRE2_VERSION_LINES}")
string(REGEX REPLACE ".*PCRE2_MINOR[ \t]+([0-9]+).*" "\\1" PCRE2_MINOR "${PCRE2_VERSION_LINES}")
if("${PCRE2_MAJOR}.${PCRE2_MINOR}" VERSION_LESS 10.45)
  message(FATAL_ERROR "PCRE2 ${PCRE2_MAJOR}.${PCRE2_MINOR} found; SPEC.md needs >= 10.45 (Unicode 16.0)")
elseif("${PCRE2_MAJOR}.${PCRE2_MINOR}" VERSION_GREATER 10.45)
  # 10.49 (Homebrew, Nov 2026) ships Unicode 17 and fails PCRE2.UnicodeVersionIsPinned + golden tests.
  message(WARNING "PCRE2 ${PCRE2_MAJOR}.${PCRE2_MINOR} found; SPEC.md pins Unicode 16.0 (PCRE2 10.45). "
                  "Build it with scripts/build_pcre2.sh ~/pcre2 and pass -DPCRE2_INCLUDE_DIR=~/pcre2/include "
                  "-DPCRE2_LIBRARY=~/pcre2/lib/libpcre2-8.a")
endif()
add_library(pcre2_8 INTERFACE)
target_include_directories(pcre2_8 INTERFACE ${PCRE2_INCLUDE_DIR})
target_link_libraries(pcre2_8 INTERFACE ${PCRE2_LIBRARY})
target_compile_definitions(pcre2_8 INTERFACE PCRE2_CODE_UNIT_WIDTH=8)

# ---- Fetched dependencies -------------------------------------------------------
include(FetchContent)
FetchContent_Declare(unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git GIT_TAG v4.4.0 GIT_SHALLOW TRUE)
FetchContent_Declare(cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git GIT_TAG v2.4.2 GIT_SHALLOW TRUE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.15.2 GIT_SHALLOW TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(unordered_dense cli11 googletest)

# ---- Common flags ---------------------------------------------------------------
add_library(parabpe_flags INTERFACE)
target_compile_options(parabpe_flags INTERFACE -Wall -Wextra -Wpedantic)
if(PARABPE_NATIVE AND NOT PARABPE_ASAN AND NOT PARABPE_TSAN)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    target_compile_options(parabpe_flags INTERFACE -march=native)
  else()
    target_compile_options(parabpe_flags INTERFACE -mcpu=native)
  endif()
endif()
# Debug: bounds-checked operator[] in libc++ / libstdc++ (ABI-compatible, so gtest needn't match).
target_compile_definitions(parabpe_flags INTERFACE
  $<$<CONFIG:Debug>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE _GLIBCXX_ASSERTIONS>)
if(PARABPE_ASAN)
  target_compile_options(parabpe_flags INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(parabpe_flags INTERFACE -fsanitize=address,undefined)
endif()
if(PARABPE_TSAN)
  target_compile_options(parabpe_flags INTERFACE -fsanitize=thread -g)
  target_link_options(parabpe_flags INTERFACE -fsanitize=thread)
endif()

# ---- Library + CLI --------------------------------------------------------------
add_library(parabpe_core
  src/corpus.cpp
  src/regex.cpp
  src/pretok_serial.cpp
  src/pretok_omp.cpp
  src/word_table.cpp
  src/merge_naive.cpp
  src/merge_incr.cpp
  src/merge_batch.cpp
  src/strategies.cpp
  src/io.cpp
  src/trace.cpp
  src/trainer.cpp
)
target_include_directories(parabpe_core PUBLIC include PRIVATE src)
target_compile_definitions(parabpe_core PRIVATE PARABPE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(parabpe_core
  PUBLIC OpenMP::OpenMP_CXX pcre2_8 unordered_dense::unordered_dense parabpe_flags)

add_executable(parabpe src/main.cpp)
target_link_libraries(parabpe PRIVATE parabpe_core CLI11::CLI11)

# ---- Tests ----------------------------------------------------------------------
enable_testing()
add_executable(parabpe_tests
  tests/unit/test_smoke.cpp
  tests/unit/test_chunker.cpp
  tests/unit/test_pretok.cpp
  tests/unit/test_unicode_parity.cpp
  tests/unit/test_golden.cpp
  tests/unit/test_io.cpp
  tests/unit/test_batch.cpp
  tests/unit/test_trace.cpp
)
target_link_libraries(parabpe_tests PRIVATE parabpe_core GTest::gtest_main)
target_include_directories(parabpe_tests PRIVATE src)
target_compile_definitions(parabpe_tests PRIVATE PARABPE_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
include(GoogleTest)
gtest_discover_tests(parabpe_tests)

# Equivalence matrix on real data (S3). Skipped (exit 77) until data/ exists; `ctest -LE slow` omits it.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME equivalence_matrix
           COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_SOURCE_DIR}/tests/equivalence/run_matrix.py
                   --input ${CMAKE_SOURCE_DIR}/data/tinystories.10M.txt --vocab-size 4000
                   --binary $<TARGET_FILE:parabpe>)
  set_tests_properties(equivalence_matrix PROPERTIES LABELS slow SKIP_RETURN_CODE 77 TIMEOUT 1800)
  # Studio server (B7): input confinement and request checks; no training runs.
  add_test(NAME studio_serve
           COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_SOURCE_DIR}/tests/studio/test_serve.py)
endif()
