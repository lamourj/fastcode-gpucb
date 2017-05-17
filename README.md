# Project #
Fast GP-UCB implementation.

## Versions ##
1. Cholesky incremental

## 1:1 Meeting May 9th ##
Todos:

- Fix kernel (to RBF)
- Neglect function cost
- Use floats (update: is it sufficient?)
- Push sizes (10, 20,... 50 iterations)

Suggestions on how to split the work:
1. Cholesky incrementing
2. Solving systems: blocking
3. Rest

Dependencies might not be too restrictives: optimization might be done by starting some computations before other end.

Use AVX2
