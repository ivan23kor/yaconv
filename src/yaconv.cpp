#include <blis.h>
namespace {
#include "blis_params.hpp" // BLIS block sizes, microkernels
};
#include "utils.hpp"  // Tensor aligned allocation and printing
#include <chrono>     // Timing
#include <cmath>      // std::ceil
#include <iostream>   // Debug printing

// Timing
using namespace std::chrono;
static high_resolution_clock::time_point t1, t2, t3, t4;
double YaConvPack1 = 0.0, YaConvPack2 = 0.0, YaConvComp = 0.0;

// TODO: Convolution parameters as a struct
static int C, H, W, M, FH, FW, OH, OW, SH, SW, PH, PW;

// Image is H x W x C
void packImage(float *Image, float *Pack, int NC) {
  for (int nr = 0; nr < NC; nr += BLOCK_NR)
    packBPanel(Image + nr * W * C, Pack + nr * W * C, MIN(NC - nr, BLOCK_NR), W * C, W * C, 1, BLOCK_NR);
}

// Filter is M x FH x FW x C
void packFilter(float *Filter, float *Pack, int MC, int KC) {
  for (int mr = 0; mr < MC; mr += BLOCK_MR)
    packAPanel(Filter + mr * FH * FW * C, Pack + mr * KC, MIN(MC - mr, BLOCK_MR), KC, FH * FW * C, 1, BLOCK_MR);
}

void yaconv(float *Image, float *Filter, float *Output,
            int C_, int H_, int W_, int M_, int FH_, int FW_,
            int PH_, int PW_, int SH_, int SW_) {

  // Output sizes
  C = C_, H = H_, W = W_, M = M_, FH = FH_, FW = FW_, PH = PH_, PW = PW_, SH = SH_, SW = SW_;
  OH = (H - FH + 2 * PH) / SH + 1;
  OW = (W - FW + 2 * PW) / SW + 1;

  // Pack Image in L3
  auto *ImagePack = alignedAlloc(C * (H + BLOCK_NR) * W);//BLOCK_MC * BLOCK_NC);
  // printTensor(Image, {H, W * C});
  // packImage(Image, ImagePack);
  // printTensor(ImagePack, {W * C * (int)std::ceil((float)H / BLOCK_NR), BLOCK_NR});

  // Pack Filter in L2
  auto *FilterPack = alignedAlloc(BLOCK_MC * BLOCK_KC);
  // printTensor(Filter, {M, FH * FW * C});
  // int MC = M, KC = 10;
  // packFilter(Filter, FilterPack, MC, KC);
  // printTensor(FilterPack, {KC * (int)std::ceil((float)M / BLOCK_MR), BLOCK_MR});

  auto *CBuff = alignedAlloc(BLOCK_MR * BLOCK_NR);

  // TODO: Add an extra loop over H; this assumes Image fits in L3, which is almost always true
  int Debug = 0;
  if (Debug == 2)
    t1 = high_resolution_clock::now();
  packImage(Image, ImagePack, H);
  if (Debug == 1) {
    printTensor(Image, {H, W * C});
    printTensor(Filter, {M, FH * FW * C});
    printTensor(ImagePack, {W * C * (int)std::ceil((float)H / BLOCK_NR), BLOCK_NR});
  }
  if (Debug == 2) {
    t2 = high_resolution_clock::now();
    YaConvPack1 += duration_cast<duration<double>>(t2 - t1).count();
    t3 = high_resolution_clock::now();
  }
  for (int fh = 0; fh < FH; ++fh) {
    for (int m = 0; m < M; m += BLOCK_MC) {
      int MC = MIN(M - m, BLOCK_MC);
      for (int kc = 0; kc < FW * C; kc += BLOCK_KC) {
        float *Beta = (fh == 0) && (kc == 0) ? bli_s0 : bli_s1;
        int KC = MIN(FW * C - kc, BLOCK_KC);
        if (Debug == 2)
          t1 = high_resolution_clock::now();
        packFilter(Filter + (m * FH + fh) * FW * C + kc, FilterPack, MC, KC);
        if (Debug == 1)
          printTensor(FilterPack, {KC * (int)std::ceil((float)MC / BLOCK_MR), BLOCK_MR});
        if (Debug == 2) {
          t2 = high_resolution_clock::now();
          YaConvPack2 += duration_cast<duration<double>>(t2 - t1).count();
        }
        for (int nr = 0; nr < H; nr += BLOCK_NR) {
          for (int ow = 0; ow < OW; ++ow) {
            int ImageStart = (ow - PW) * C + kc;
            int ImageEnd = MIN(W * C, ImageStart + KC);

            float *Ar = FilterPack;
            if (ImageStart < 0) {
              Ar -= ImageStart * BLOCK_MR;
              ImageStart = 0;
            }

            int K = ImageEnd - ImageStart;
            if (Debug == 1)
              std::cout << "fh = " << fh << ", ImageStart = " << ImageStart << ", K = " << K << ", OutputOff = " << ((nr - fh + PH) * OW + ow) * M + m<< "\n";
            if (K <= 0)
              continue;

            float *Br = ImagePack + nr * W * C + ImageStart * BLOCK_NR;
            float *Cr = Output + ((nr - fh + PH) * OW + ow) * M + m;

            for (int mr = 0; mr < MC; mr += BLOCK_MR) {
              if (Debug == 1) {
                std::cout << "Going to gemm Weight:\n";
                printTensor(Ar, {K, BLOCK_MR});
                std::cout << "and Image" << std::endl;
                printTensor(Br, {K, BLOCK_NR});
              }
              if (mr + BLOCK_MR <= MC) {
                bli_sset0s_mxn(1, BLOCK_MR, Cr, 1, OW * M);
                sgemm_ukr(K, bli_s1, Ar, Br, Beta, Cr, 1, OW * M);
              } else {
                sgemm_ukr(K, bli_s1, Ar, Br, bli_s0, CBuff, BLOCK_NR, 1);
                bli_sxpbys_mxn(MC - mr, BLOCK_NR, CBuff, BLOCK_NR, 1, Beta, Cr, 1, OW * M);
              }
              if (Debug == 1) {
                std::cout << "so now the result is:" << std::endl;
                printTensor(Output, {OH * OW, M});
              }

              Ar += BLOCK_MR * KC;
              Cr += BLOCK_MR;
            }
          }
        }
      }
    }
  }
  if (Debug == 2) {
    t4 = high_resolution_clock::now();
    YaConvComp += duration_cast<duration<double>>(t4 - t3).count();
  }
}
