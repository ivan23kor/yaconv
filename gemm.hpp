#pragma once

void packA(const float *Block, float *&Pack, unsigned LDA, unsigned MC,
           unsigned KC);

void packB(const float *Block, float *&Pack, unsigned LDB, unsigned KC,
           unsigned NC);

void gemm(float *A, float *B, float *C, unsigned M, unsigned K,
          unsigned N, unsigned LDA, unsigned LDB, unsigned LDC, float Alpha,
          float Beta);
