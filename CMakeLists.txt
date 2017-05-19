cmake_minimum_required(VERSION 3.6)
project(fastcode-gpucb)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES
        src/gpucb3.h
        src/gpucb3.c
        src/mathHelpers3.h
        src/mathHelpers3.c
        src/main.c
        src/main.h
        )
add_executable(fastcode-gpucb ${SOURCE_FILES})

## Compiler flags
set(CMAKE_C_FLAGS "-O3 -march=core-avx2")        ## Optimize