#include "utils.hpp"
#include <blis.h>
#include <iostream>
#include <stdlib.h>

namespace {
#include "blis_utils.hpp"
}; // namespace

// A is row-major
void packA(float *A, float *Pack, int LDA, int MC, int KC) {
  int ic = 0;
  for (; ic + BLOCK_MR <= MC; ic += BLOCK_MR)
    packAPanel(A + ic * LDA, Pack + ic * KC, BLOCK_MR, KC, LDA, 1, BLOCK_MR);
  packAPanel(A + ic * LDA, Pack + ic * KC, MC - ic, KC, LDA, 1, BLOCK_MR);
}

// B is row-major
void packB(float *B, float *Pack, int LDB, int KC, int NC) {
  int jc = 0;
  for (; jc + BLOCK_NR < NC; jc += BLOCK_NR)
    packBPanel(B + jc, Pack + jc * KC, BLOCK_NR, KC, 1, LDB, BLOCK_NR);
  packBPanel(B + jc, Pack + jc * KC, NC - jc, KC, 1, LDB, BLOCK_NR);
}

static float *APack = nullptr, *BPack = nullptr, *CBuff = nullptr;

void initGemmBuffers() {
  APack = alignedAlloc(BLOCK_MC * BLOCK_KC);
  BPack = alignedAlloc(BLOCK_KC * BLOCK_NC);
  CBuff = alignedAlloc(BLOCK_MR * BLOCK_NR);
};

void freeGemmBuffers() {
  free(APack);
  free(BPack);
  free(CBuff);
}

// TODO: As compared to BLIS, this gemm can perform +10% or -10%, depending
// on the input matrix sizes
// TODO: As compared to OpenBLAS, this gemm usually performs as worse
// as BLIS does (-10%); but OpenBLAS is 2x-3x faster for skinny GEMMs (K = 3)
void gemm(float *A, float *B, float *C, int M, int K, int N,
          int LDA, int LDB, int LDC, float Alpha, float Beta) {

  initGemmBuffers();

  float Zero = 0.0, One = 1.0;
  for (int jc = 0; jc < N; jc += BLOCK_NC) {

    int NC = MIN(N - jc, BLOCK_NC);

    for (int k = 0; k < K; k += BLOCK_KC) {

      int KC = MIN(K - k, BLOCK_KC);
      float Beta_ = k == 0 ? Beta : One;

      packB(B + k * LDB + jc, BPack, LDB, KC, NC);

      for (int ic = 0; ic < M; ic += BLOCK_MC) {

        int MC = MIN(M - ic, BLOCK_MC);

        packA(A + ic * LDA + k, APack, LDA, MC, KC);

        for (int jr = 0; jr < NC; jr += BLOCK_NR) {

          int NR = MIN(NC - jr, BLOCK_NR);

          for (int ir = 0; ir < MC; ir += BLOCK_MR) {

            int MR = MIN(MC - ir, BLOCK_MR);

            float *Ar = APack + ir * KC;
            float *Br = BPack + jr * KC;
            float *Cr = C + (ic + ir) * LDC + jc + jr;

            // Cr = Beta * Cr + Alpha * Ar x Br
            if ((MR == BLOCK_MR) && (NR == BLOCK_NR))
              sgemm_ukr(KC, &Alpha, Ar, Br, &Beta_, Cr, LDC, 1);
            else {
              sgemm_ukr(KC, &Alpha, Ar, Br, &Zero, CBuff, BLOCK_NR, 1);
              bli_sxpbys_mxn(MR, NR, CBuff, BLOCK_NR, 1, &Beta_, Cr, LDC, 1);
            }
          }
        }
      }
    }
  }
  freeGemmBuffers();
}
