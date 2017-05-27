cmake_minimum_required(VERSION 3.6)
project(fastcode-gpucb)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES
        src/main.c
        src/main.h
        src/gpucb8.c
        src/gpucb8.h
        src/avx_mathfun.h)
add_executable(fastcode-gpucb ${SOURCE_FILES})

## Compiler flags.
set(CMAKE_C_FLAGS "-O3 -march=core-avx2 -fno-tree-vectorize")        ## Optimize