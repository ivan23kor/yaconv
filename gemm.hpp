#pragma once

void gemm(float *A, float *B, float *C, int M, int K,
          int N, int LDA, int LDB, int LDC, float Alpha,
          float Beta);
