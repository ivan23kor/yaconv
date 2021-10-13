#pragma once

void gemm(float *A, float *B, float *C, unsigned M, unsigned K,
          unsigned N, unsigned LDA, unsigned LDB, unsigned LDC, float Alpha,
          float Beta);
