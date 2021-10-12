#include "utils.hpp"
#include <blis.h>
#include <iostream>
#include <stdlib.h>

namespace {
#include "blis_params.hpp"
}; // namespace

// TODO: hopefully, the compiler is doing if-unswitching here.
// It is ok for now as it keeps the code clean.
// Anyway, manually unswitched packing was slower than that from BLIS so
// packing performance optimization must be left for later

// Block is row-major
void packA(const float *Block, float *&Pack, unsigned LDA, unsigned MC,
           unsigned KC) {
  unsigned To = 0;
  for (unsigned ic = 0; ic < MC; ic += BLOCK_MR) {
    unsigned MR = MIN(MC - ic, BLOCK_MR);
    for (unsigned k = 0; k < KC; ++k) {
      for (unsigned ir = 0; ir < MR; ++ir) {
        unsigned From = (ic + ir) * LDA + k;
        Pack[To++] = Block[From];
      }
      To += BLOCK_MR - MR;
    }
  }
}

// Block is row-major
void packB(const float *Block, float *&Pack, unsigned LDB, unsigned KC,
           unsigned NC) {
  unsigned To = 0;
  for (unsigned jc = 0; jc < NC; jc += BLOCK_NR) {
    unsigned NR = MIN(NC - jc, BLOCK_NR);
    for (unsigned k = 0; k < KC; ++k) {
      for (unsigned jr = 0; jr < NR; ++jr) {
        unsigned From = jc + jr + LDB * k;
        Pack[To++] = Block[From];
      }
      To += BLOCK_NR - NR;
    }
  }
}

void gemm(const float *A, const float *B, float *C, unsigned M, unsigned K,
          unsigned N, unsigned LDA, unsigned LDB, unsigned LDC, float Alpha,
          float Beta) {

  auto *APack = alignedAlloc(BLOCK_MC * BLOCK_KC);
  auto *BPack = alignedAlloc(BLOCK_KC * BLOCK_NC);
  auto *CBuff = alignedAlloc(BLOCK_MR * BLOCK_NR);

  float Zero = 0.0, One = 1.0;
  for (unsigned jc = 0; jc < N; jc += BLOCK_NC) {

    unsigned NC = MIN(N - jc, BLOCK_NC);

    for (unsigned k = 0; k < K; k += BLOCK_KC) {

      unsigned KC = MIN(K - k, BLOCK_KC);

      packB(B + k * LDB + jc, BPack, LDB, KC, NC);

      for (unsigned ic = 0; ic < M; ic += BLOCK_MC) {

        unsigned MC = MIN(M - ic, BLOCK_MC);

        packA(A + ic * LDA + k, APack, LDA, MC, KC);

        for (unsigned jr = 0; jr < NC; jr += BLOCK_NR) {

          unsigned NR = MIN(NC - jr, BLOCK_NR);

          for (unsigned ir = 0; ir < MC; ir += BLOCK_MR) {

            unsigned MR = MIN(MC - ir, BLOCK_MR);

            float *Ar = APack + ir * KC;
            float *Br = BPack + jr * KC;
            float *Cr = C + (ic + ir) * LDC + jc + jr;

            // Cr = Beta * Cr + Alpha * Ar x Br
            if ((MR == BLOCK_MR) && (NR == BLOCK_NR))
              bli_sgemm_ukr(KC, &Alpha, Ar, Br, &Beta_, Cr, LDC, 1, data, cntx);
            else {
              bli_sgemm_ukr(KC, &Alpha, Ar, Br, &Zero, CBuff, BLOCK_NR, 1, data, cntx);
              bli_sxpbys_mxn(MR, NR, CBuff, BLOCK_NR, 1, &Beta_, Cr, LDC, 1);
            }
          }
        }
      }
    }
  }
}
