# Project

## Compilation ##
After having installed with brew:

gcc -Wall -O3 -fno-tree-vectorize -I/usr/local/include -c src/main.c -o src/main.o
gcc -O3 -fno-tree-vectorize -L/usr/local/lib -lgsl -lgslcblas src/main.o