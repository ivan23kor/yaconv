#include "utils.hpp"  // Tensor aligned allocation and printing
#include <blis.h>     // BLIS microkernel and block sizes
#include <chrono>     // Timing
#include <iostream>

using namespace std::chrono;
static high_resolution_clock::time_point t1, t2;

namespace {
#include "blis_params.hpp"
}; // namespace

// Defined in test_conv.cpp (driver code for convolutions)
// Algorithms in this file will append times to this vector
extern std::vector<double> Times;

void packInput(const float *Input, float *Pack,
    unsigned C, unsigned H, unsigned W, unsigned KW,
    unsigned MC, unsigned KC, unsigned MS, unsigned KS) {

  unsigned To = 0;
  for (unsigned ic = MS; ic < MS + MC; ic += BLOCK_MR) {
    unsigned MR = MIN(MS + MC - ic, BLOCK_MR);
    for (unsigned k = KS; k < KS + KC; ++k) {
      for (unsigned ir = 0; ir < BLOCK_MR; ++ir) {
        unsigned c = k % C;
        unsigned h = k / C / KW;
        unsigned w = k / C % KW + ic + ir;
        unsigned From = c * H * W + h * W + w;
        Pack[To++] = ir < MR ? Input[From] : 0.0;
      }
    }
  }
}

void packKernel(const float *Kernel, float *Pack,
    unsigned C, unsigned KH, unsigned KW,
    unsigned KC, unsigned NC, unsigned KS, unsigned NS) {

  unsigned To = 0;
  for (unsigned jc = NS; jc < NS + NC; jc += BLOCK_NR) {
    unsigned NR = MIN(NS + NC - jc, BLOCK_NR);
    for (unsigned k = KS; k < KS + KC; ++k) {
      for (unsigned jr = 0; jr < BLOCK_NR; ++jr) {
        unsigned m = jc + jr;
        unsigned c = k % C;
        unsigned h = k / C / KW;
        unsigned w = k / C % KW;
        unsigned From = m * C * KH * KW + c * KH * KW + h * KW + w;
        Pack[To++] = jr < NR ? Kernel[From] : 0.0;
      }
    }
  }
}

float *yaconv(const float *Input, float *Kernel, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
              unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
              unsigned StrideH, unsigned StrideW, unsigned DilH, unsigned DilW) {

  unsigned BLOCK_OW = OW + BLOCK_MR - (OW - 1) % BLOCK_MR - 1;
  unsigned BLOCK_M = M + BLOCK_NR - (M - 1) % BLOCK_NR - 1;

  TIME(
  auto *Output = alignedAlloc(M * OH * OW);
  auto *KernelPack = alignedAlloc(BLOCK_M * KH * KW * C);
  auto *InputPack = alignedAlloc(H * KW * C * BLOCK_OW);
  )

  TIME(packInput(Input, InputPack, C, H, W, KW, OW, H * KW * C, 0, 0);)
  TIME(packKernel(Kernel, KernelPack, C, KH, KW, KH * KW * C, M, 0, 0);)

  std::cout << "Packed Input:\n";
  printTensor(InputPack, {H * KW * C * BLOCK_OW / BLOCK_MR, BLOCK_MR});
  std::cout << "Packed Kernel:\n";
  printTensor(KernelPack, {KH * KW * C * BLOCK_M / BLOCK_NR, BLOCK_NR});
  return Output;

  TIME(packInput(Input, InputPack, C, H, W, KW, OW, H * KW * C, 0, 0);)
  TIME(packKernel(Kernel, KernelPack, C, KH, KW, KH * KW * C, M, 0, 0);)

  float Zero = 0.0, One = 1.0, Alpha = One;
  auto *CTile = new float[BLOCK_MR * BLOCK_NR];
  // std::cout << "Yaconv GEMMs: " << OH << " \"" << BLOCK_M << " x " << KH * KW * C << " x " << BLOCK_OW << "\"\n";

  TIME(
  for (unsigned ir = 0; ir < M; ir += BLOCK_MR) {

    unsigned MR = MIN(M - ir, BLOCK_MR);

    for (unsigned jr = 0; jr < OW; jr += BLOCK_NR) {

      unsigned NR = MIN(OW - jr, BLOCK_NR);

      for (unsigned h = 0; h < OH; ++h) {

        for (unsigned kc = 0; kc < KH * KW * C; kc += BLOCK_KC) {

          auto *Ar = KernelPack + ir * KH * KW * C + kc * BLOCK_MR;
          auto *Br = InputPack + jr * H * KW * C + h * KW * C * BLOCK_NR + kc * BLOCK_NR;
          auto *Cr = Output + ir * OH * OW + h * OW + jr;

          unsigned KC = MIN(KH * KW * C - kc, BLOCK_KC);
          // printTensor(Ar, {KC, BLOCK_MR});
          // printTensor(Br, {KC, BLOCK_NR});

          float Beta = kc == 0 ? Zero : One;
          if ((MR == BLOCK_MR) && (NR == BLOCK_NR))
            // Full tiles
            blisGemmUKR(KC, &Alpha, Ar, Br, &Beta, Cr, OH * OW, 1, data, cntx);
          else {
            // Remainder tiles
            blisGemmUKR(KC, &Alpha, Ar, Br, &Beta, CTile, BLOCK_NR, 1, data, cntx);
            bli_scopym(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE, MR, NR, CTile, BLOCK_NR, 1, Cr, OH * OW, 1);
          }
        }
      }
    }
  }
  )

  return Output;
}
