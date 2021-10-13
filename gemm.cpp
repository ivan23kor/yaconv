#include "utils.hpp"
#include <blis.h>
#include <iostream>
#include <stdlib.h>

namespace {
#include "blis_params.hpp"
}; // namespace

// Helper functions for packing, hide some BLIS parameters for panel packing
inline void packAPanel(float *APanel, float *PackPanel, unsigned MR, unsigned KC, unsigned rsa, unsigned csa, unsigned incp) {
  bli_packA_ukr(BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS, MR, KC, KC, bli_s1, APanel, rsa, csa, PackPanel, incp, cntx);
};

inline void packBPanel(float *BPanel, float *PackPanel, unsigned NR, unsigned KC, unsigned rsb, unsigned csb, unsigned incp) {
  bli_packB_ukr(BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS, NR, KC, KC, bli_s1, BPanel, rsb, csb, PackPanel, incp, cntx);
};

// A is row-major
void packA(float *A, float *&Pack, unsigned LDA, unsigned MC, unsigned KC) {
  unsigned ic = 0;
  for (; ic + BLOCK_MR <= MC; ic += BLOCK_MR)
    packAPanel(A + ic * LDA, Pack + ic * KC, BLOCK_MR, KC, LDA, 1, BLOCK_MR);
  packAPanel(A + ic * LDA, Pack + ic * KC, MC - ic, KC, LDA, 1, BLOCK_MR);
}

// B is row-major
void packB(float *B, float *&Pack, unsigned LDB, unsigned KC, unsigned NC) {
  unsigned jc = 0;
  for (; jc + BLOCK_NR < NC; jc += BLOCK_NR)
    packBPanel(B + jc, Pack + jc * KC, BLOCK_NR, KC, 1, LDB, BLOCK_NR);
  packBPanel(B + jc, Pack + jc * KC, NC - jc, KC, 1, LDB, BLOCK_NR);
}

// TODO: As compared to BLIS, this gemm can perform +10% or -10%, depending
// on the input matrix sizes
// TODO: As compared to OpenBLAS, this gemm usually performs as worse
// as BLIS does (-10%); but OpenBLAS is 2x-3x faster for skinny GEMMs (K = 3)
void gemm(float *A, float *B, float *C, unsigned M, unsigned K,
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
      float Beta_ = k == 0 ? Beta : One;

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
