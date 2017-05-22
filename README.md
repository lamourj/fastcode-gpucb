# Project #
Fast GP-UCB implementation.

## Versions ##
Baseline now has incremental Cholesky.
0. Baseline without incremental Cholesky
1. ~~Incremental Cholesky, now baseline~~
2. ~~First optimization without vectorization, had incoherences: deleted~~
3. Vectorized search of the maximum value to sample.

## 1:1 Meeting May 9th ##
Todos:

- Fix kernel (to RBF) -- done
- Neglect function cost -- done?
- Use floats (update: is it sufficient?)
- Push sizes (10, 20,... 50 iterations)

Suggestions on how to split the work:
1. Cholesky incrementing
2. Solving systems: blocking
3. Rest

Dependencies might not be too restrictives: optimization might be done by starting some computations before other end.

Use AVX2
