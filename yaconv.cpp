#include <blis.h>
namespace {
#include "blis_params.hpp" // BLIS block sizes, microkernels
};
#include "utils.hpp"  // Tensor aligned allocation and printing
#include <chrono>     // Timing
#include <iostream>   // Debug printing

// Timing
using namespace std::chrono;
static high_resolution_clock::time_point t1, t2, t3, t4;
double YaConvPack1 = 0.0, YaConvPack2 = 0.0, YaConvComp = 0.0;

// TODO: Convolution parameters as a struct
static int C, H, W, M, FH, FW, OH, OW, SH, SW, PH, PW;

inline bool yaconvCanHandle() {
  return !((SH != 1)
      // Haven't figured out how to slice bigger matrices yet
      || (C > BLOCK_KC)
      // Image must fit in TLB/L2
      || (std::ceil((float)H / BLOCK_MR) * BLOCK_MR * W * C > BLOCK_MC * BLOCK_KC)
      // Filter must fit in L3
      || (std::ceil((float)M / BLOCK_NR) * BLOCK_NR * FH * FW * C > BLOCK_KC * BLOCK_NC));
}

// Assumes Image is H x W x C
void packImage(float *Image, float *Pack) {
  int mr = 0;
  for(; mr + BLOCK_MR <= H; mr += BLOCK_MR)
    packAPanel(Image + mr * W * C, Pack + mr * W * C, BLOCK_MR, W * C, W * C, 1, BLOCK_MR);
  packAPanel(Image + mr * W * C, Pack + mr * W * C, H - mr, W * C, W * C, 1, BLOCK_MR);
}

// Assumes Filter is M x FH x FW x C
void packFilter(float *Filter, float *Pack) {
  int WholeM = M % BLOCK_NR == 0 ? M : M + BLOCK_NR - M % BLOCK_NR;

  int K = FH * FW * C;
  for (int f = 0; f < K; f += C) {
    int nr = 0;
    for (; nr + BLOCK_NR <= M; nr += BLOCK_NR)
      packBPanel(Filter + f + nr * K, Pack + f * WholeM + nr * C, BLOCK_NR, C, K, 1, BLOCK_NR);
    packBPanel(Filter + f + nr * K, Pack + f * WholeM + nr * C, M - nr, C, K, 1, BLOCK_NR);
  }
}

void yaconv(float *Image, float *Filter, float *Output,
            int C_, int H_, int W_, int M_, int FH_, int FW_,
            int PH_, int PW_, int SH_, int SW_) {

  // Output sizes
  C = C_, H = H_, W = W_, M = M_, FH = FH_, FW = FW_, PH = PH_, PW = PW_, SH = SH_, SW = SW_;
  OH = (H - FH + 2 * PH) / SH + 1;
  OW = (W - FW + 2 * PW) / SW + 1;

  // Only unit strides, for other strides im2col has less overhead anyway
  if (!yaconvCanHandle()) {
    std::cout << "\033[31myaconv con't handle such convolution yet\033[0m\n";
    exit(-1);
  }

  // Initialize all Output elements to zero
  bli_sset0s_mxn(M, OH * OW, Output, OH * OW, 1);

  // Pack Filter in L3
  t1 = high_resolution_clock::now();
  auto *FilterPack = alignedAlloc(BLOCK_KC * BLOCK_NC);
  packFilter(Filter, FilterPack);
  t2 = high_resolution_clock::now();
  YaConvPack1 += duration_cast<duration<double>>(t2 - t1).count();
  // printTensor(Filter, {M, FH* FW* C});
  // printTensor(FilterPack, {FH * FW * C * (int)std::ceil((float)M / BLOCK_NR), BLOCK_NR});

  // Pack Image in L2
  t1 = high_resolution_clock::now();
  auto *ImagePack = alignedAlloc(BLOCK_MC * BLOCK_KC);
  packImage(Image, ImagePack);
  t2 = high_resolution_clock::now();
  YaConvPack2 += duration_cast<duration<double>>(t2 - t1).count();
  // printTensor(Image, {H, W * C});
  // printTensor(ImagePack, {W * C * (int)std::ceil((float)H / BLOCK_MR), BLOCK_MR});

  t1 = high_resolution_clock::now();
  int WholeM = M % BLOCK_NR == 0 ? M : M + (BLOCK_NR - M % BLOCK_NR);
  int WholeH = H % BLOCK_MR == 0 ? H : H + (BLOCK_MR - H % BLOCK_MR);
  int MinOff = 0, MaxOff = 0;
  Output += (PH * OW + PW) * M;
  for (int f = 0; f < FH * FW; ++f) {
    int fh = f / FW, fw = f % FW;
    int FromRow = MAX(0, fh - PH);
    int UntilRow =  MIN(H, fh + H + PH - FH + 1);
    int FromCol = MAX(0, fw - PW);
    int UntilCol =  MIN(W, fw + W + PW - FW + 1);
    for (int m = 0; m < M; m += BLOCK_NR) {
      for (int w = FromCol; w < UntilCol; ++w) {
        for (int h = 0; h < H; h += BLOCK_MR) {
          float *Ar = ImagePack + w * BLOCK_MR * C + h * W * C;
          float *Br = FilterPack + f * WholeM * C + m * C;
          // printTensor(Ar, {C, BLOCK_MR});
          // printTensor(Br, {C, BLOCK_NR});
          int off = ((h - fh) * OW + w - fw) * M + m;
          // std::cout << f << "(f), " << m << "(m), " << w << "(w), " << h << "(h), " << off << "(off) " << FromRow << " -> " << UntilRow << "\n";
          MinOff = MIN(MinOff, off);
          MaxOff = MAX(MaxOff, off);
          float *Cr = Output + off;
          sgemm_ukr(C, bli_s1, Ar, Br, bli_s1, Cr, OW * M, 1);
          // printTensor(Output, {OH * OW, M});
        }
      }
    }
  }
  t2 = high_resolution_clock::now();
  YaConvComp += duration_cast<duration<double>>(t2 - t1).count();
  return;
  int MyMinOff = 0 - ((FH - 1) * OW + PW) * M;
  if (MinOff != MyMinOff)
    std::cout << "Min: " << MinOff  << " != " << MyMinOff << "\n";
  // MaxOff contains the offset of the beginning of the farthest row.
  // After this, MaxOff will contain the beginning of the next row
  MaxOff += BLOCK_MR * OW * M + BLOCK_NR;
  int MyMaxOff = WholeH * OW * M + (W + PW - FW + 1) * M;
  if (MaxOff != MyMaxOff)
    std::cout << "Max: " << MaxOff  << " != " << MyMaxOff << "\n";
}
