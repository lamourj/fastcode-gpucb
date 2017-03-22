cmake_minimum_required(VERSION 3.6)
project(fastcode-gpucb)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c)
add_executable(fastcode-gpucb ${SOURCE_FILES})