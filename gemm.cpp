#include <blis.h>
#include <stdlib.h>

#define MIN(a, b) a < b ? a : b

namespace {

// BLIS-related stuff
auto *cntx = bli_gks_query_cntx();
auto *data = new auxinfo_t;

unsigned BLOCK_MR = bli_cntx_get_blksz(BLIS_MR, cntx)->v[0];
unsigned BLOCK_NR = bli_cntx_get_blksz(BLIS_NR, cntx)->v[0];
unsigned BLOCK_MC = bli_cntx_get_blksz(BLIS_MC, cntx)->v[0];
unsigned BLOCK_KC = bli_cntx_get_blksz(BLIS_KC, cntx)->v[0];
unsigned BLOCK_NC = bli_cntx_get_blksz(BLIS_NC, cntx)->v[0];

}; // abstract namespace

// Block is row-major
void packA(float *Block, float *&Pack, unsigned LDA, unsigned MC, unsigned KC) {
  unsigned To = 0;
  for (unsigned ic = 0; ic < MC; ic += BLOCK_MR) {
    unsigned MR = MIN(MC - ic, BLOCK_MR);
    for (unsigned k = 0; k < KC; ++k) {
      for (unsigned ir = 0; ir < MR; ++ir) {
        unsigned From = (ic + ir) * LDA + k;
        Pack[To++] = Block[From];
      }
      for (unsigned End = To + BLOCK_MR - MR; To != End; ++To)
        Pack[To] = 0.0;
    }
  }
}

// Block is row-major
void packB(float *Block, float *&Pack, unsigned LDB, unsigned KC, unsigned NC) {
  unsigned To = 0;
  for (unsigned jc = 0; jc < NC; jc += BLOCK_NR) {
    unsigned NR = MIN(NC - jc, BLOCK_NR);
    for (unsigned k = 0; k < KC; ++k) {
      for (unsigned jr = 0; jr < NR; ++jr) {
        unsigned From = jc + jr + LDB * k;
        Pack[To++] = Block[From];
      }
      for (unsigned End = To + BLOCK_NR - NR; To != End; ++To)
        Pack[To] = 0.0;
    }
  }
}

void gemm(float *A, float *B, float *C, unsigned M, unsigned K, unsigned N,
          unsigned LDA, unsigned LDB, unsigned LDC, float Alpha, float Beta) {

  auto *APack = (float *)aligned_alloc(4096, BLOCK_MC * BLOCK_KC * sizeof(float));
  auto *BPack = (float *)aligned_alloc(4096, BLOCK_KC * BLOCK_NC * sizeof(float));
  auto *CBuff = (float *)aligned_alloc(4096, BLOCK_MR * BLOCK_NR * sizeof(float));

  float Zero = 0.0;
  for (unsigned jc = 0; jc < N; jc += BLOCK_NC) {

    unsigned NC = MIN(N - jc, BLOCK_NC);

    for (unsigned k = 0; k < K; k += BLOCK_KC) {

      unsigned KC = MIN(K - k, BLOCK_KC);

      float Beta_ = k == 0 ? Beta : 1.0; // Accumulate or not

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

            if ((MR == BLOCK_MR) && (NR == BLOCK_NR))
              bli_sgemm_haswell_asm_6x16(KC, &Alpha, Ar, Br, &Beta_,
                  Cr, LDC, 1, data, cntx);
            else {
              bli_sgemm_haswell_asm_6x16(KC, &Alpha, Ar, Br, &Zero,
                  CBuff, BLOCK_NR, 1, data, cntx);
              bli_saxpym(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
                  MR, NR, &Alpha, CBuff, BLOCK_NR, 1, Cr, LDC, 1);
            }
          }
        }
      }
    }
  }
}
