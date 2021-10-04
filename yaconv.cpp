#include "gemm.hpp"  // Tensor aligned allocation and printing
#include "utils.hpp"  // Tensor aligned allocation and printing
#include "yaconv.hpp" // class Indexer
#include <blis.h>     // BLIS microkernel and block sizes
#include <chrono>     // Timing
#include <iostream>   // Debug printing
#include <set>        // Group construction

using namespace std::chrono;
static high_resolution_clock::time_point t1, t2;

namespace {
#include "blis_params.hpp"
}; // namespace

// Defined in test_conv.cpp (driver code for convolutions)
// Algorithms in this file will append times to this vector
extern std::vector<double> Times;

void packImage(const float *Image, float *Pack, unsigned MS, unsigned MC,
    unsigned C, unsigned H, unsigned W) {

  unsigned To = 0;
  for (unsigned ic = MS; ic < MS + MC; ic += BLOCK_MR) {
    unsigned MR = MIN(MS + MC - ic, BLOCK_MR);
    for (unsigned c = 0; c < C; ++c) {
      for (unsigned ir = 0; ir < MR; ++ir) {
        unsigned From = c * H * W + ic + ir;
        Pack[To++] = Image[From];
      }
      To += BLOCK_MR - MR;
    }
  }
}

void packFilter(const float *Filter, float *Pack, unsigned NS, unsigned NC,
    unsigned M, unsigned C, unsigned FH, unsigned FW) {

  unsigned To = 0;
  for (unsigned jc = NS; jc < NS + NC; jc += BLOCK_NR) {
    unsigned NR = MIN(NS + NC - jc, BLOCK_NR);
    for (unsigned c = 0; c < C; ++c) {
      for (unsigned jr = 0; jr < NR; ++jr) {
        unsigned m = (jc + jr) % M;
        unsigned fh = (jc + jr) / M / FW;
        unsigned fw = (jc + jr) / M % FW;
        unsigned From = c * FH * FW + m * C * FH * FW + fh * FW + fw;
        Pack[To++] = Filter[From];
      }
      To += BLOCK_NR - NR;
    }
  }
}

// Assumptions: stride == 1
void packImageCenter(float *ImageOff, float *Pack, unsigned MS, unsigned MC,
    unsigned C, unsigned H, unsigned W, unsigned CenterW, unsigned SkipW) {

  for (unsigned i = 0; i < MC; ++i) {
    unsigned ImageIdx = i + i / CenterW * SkipW;
    unsigned To = (i - i % BLOCK_MR) * C + i % BLOCK_MR;
    for (unsigned c = 0, From = ImageIdx; c < C; ++c, From += H * W) {
      Pack[To++] = ImageOff[From];
    }
  }
}

void yaconv(float *Image, float *Filter, float *Output, unsigned C,
            unsigned H, unsigned W, unsigned M, unsigned FH, unsigned FW,
            unsigned SH, unsigned SW, unsigned PH, unsigned PW) {

  // TODO: remove this when non-unit stride becomes supported
  if ((SH != 1) || (SW != 1)) {
    std::cout << "\033[31mSorry, only accepting unit stride convolutions for the moment\033[0m\n";
    exit(-1);
  }

  // Output sizes
  const unsigned OH = (H - FH + 2 * PH) / SH + 1;
  const unsigned OW = (W - FW + 2 * PW) / SW + 1;

  // Center gaps, sizes
  unsigned GapW = FW - 1 - PW, GapH = FH - 1 - PH;
  unsigned CenterW = W - 2 * GapW, CenterH = H - 2 * GapH;
  unsigned CenterSize = CenterH * CenterW;
  unsigned Off = W * (FH - 1 - PH) + GapW;
  // std::cout << 100. * CenterSize / H / W << "% of the image\n";

  // GEMM block sizes
  BLOCK_MC *= BLOCK_KC / C;
  BLOCK_NC *= BLOCK_KC / C;
  // BLOCK_MC = BLOCK_MR * 2;
  // BLOCK_NC = BLOCK_NR * 2;
  BLOCK_KC = C;
  // std::cout << "Block sizes:"
  //   << " MC = " << BLOCK_MC
  //   << ", KC = " << BLOCK_KC
  //   << ", NC = " << BLOCK_NC
  //   << ", MR = " << BLOCK_MR
  //   << ", NR = " << BLOCK_NR
  //   << "\n";

  // Buffers
  auto *ImagePack = alignedAlloc(BLOCK_MC * BLOCK_KC);
  auto *FilterPack = alignedAlloc(BLOCK_KC * BLOCK_NC);
  auto *TmpOutput = alignedAlloc(BLOCK_MR * BLOCK_NR);

  // Timing
  high_resolution_clock::time_point p1, p2, p3, p4;
  double PackImageTime = 0.0, PackFilterTime = 0.0;

  // GEMM alpha, beta
  float Alpha = 1.0, One = 1.0;

  // NC loop
  for (unsigned nc = 0; nc < M * FH * FW; nc += BLOCK_NC) {

    unsigned NC = MIN(M * FH * FW - nc, BLOCK_NC);

    // p1 = high_resolution_clock::now();
    packFilter(Filter, FilterPack, nc, NC, M, C, FH, FW);
    // p2 = high_resolution_clock::now();
    // PackFilterTime += duration_cast<duration<double>>(p2 - p1).count();
    // printTensor(FilterPack, {C * (unsigned)std::ceil((float)NC / (float)BLOCK_NR), BLOCK_NR});

    for (unsigned mc = 0; mc < CenterSize; mc += BLOCK_MC) {

      unsigned MC = MIN(CenterSize - mc, BLOCK_MC);

      p3 = high_resolution_clock::now();
      packImageCenter(Image + Off, ImagePack, mc, MC, C, H, W, CenterW, 2 * GapW);
      p4 = high_resolution_clock::now();
      PackImageTime += duration_cast<duration<double>>(p4 - p3).count();
      IF_DEBUG(printTensor(ImagePack, {C * (unsigned)std::ceil((float)MC / (float)BLOCK_MR), BLOCK_MR});)

      for (unsigned nr = 0; nr < NC; nr += BLOCK_NR) {

        unsigned NR = MIN(NC - nr, BLOCK_NR);

        for (unsigned mr = 0; mr < MC; mr += BLOCK_MR) {

          unsigned MR = MIN(MC - mr, BLOCK_MR);

          // std::cout << "======================================================\n";
          // printTensor(ImagePack + mr * C, {C, BLOCK_MR});
          // std::cout << "\n\t*\n\n";
          // printTensor(FilterPack + nr * C, {C, BLOCK_NR});
          // std::cout << "\n\t=\n\n";

          // Microkernel call
          blisGemmUKR(C, &Alpha, ImagePack + mr * C, FilterPack + nr * C, &One, TmpOutput, BLOCK_NR, 1, data, cntx);

          // Copy to correct output location
          // unsigned fh = (nc + nr) / M / FW, fw = (nc + nr) / M % FW;
          // std::cout << "\033[32mfh = " << fh << ", fw = " << fw << "\033[0m\n";

          // printTensor(TmpOutput, {BLOCK_MR, BLOCK_NR});
          // std::cout << "\n\t-\n\n";
          // std::cout << "There are " << BLOCK_MR << " x " << BLOCK_NR << " elements.\n";
          // for (unsigned row = mc + mr; row < mc + mr + MR; ++row) {
          //   std::cout << "Row[" << row << "] \n";
          //   for (unsigned col = nc + nr; col < nc + nr + NR; ++col) {
          //     unsigned m = col % M;
          //     unsigned fh = col / M / FW;
          //     unsigned fw = col / M % FW;
          //     unsigned pos = W * (FH - 1 - PH) + GapW + row / CenterW ;
          //     std::cout << col << "(" << m << ", " << fh << ", " << fw << ", " << pos << ") ";
          //   }
          //   std::cout << "\n";
          // }
          // std::cout << "======================================================\n\n";
        }
      }
    }
  }
  // Times.push_back(PackFilterTime);
  Times.push_back(PackImageTime);
}

  // // Compute groups
  // for (unsigned h = 0; h < H; ++h) {
  //   for (unsigned w = 0; w < W; ++w) {
  //     unsigned ImageIdx = h * W + w;
  //     std::set<unsigned> FilterSet;
  //     for (unsigned kh = 0; kh < FH; ++kh) {
  //       for (unsigned kw = 0; kw < FW; ++kw) {
  //         unsigned FilterIdx = kh * FW + kw;
  //         if    ((h + PH >= kh) && (h + (FH - 1 - kh) < H + PH)   // height within padded area
  //             && (w + PW >= kw) && (w + (FW - 1 - kw) < W + PW))  // width within padded area
  //           FilterSet.insert(FilterIdx);
  //       }
  //     }
  //   }
  // }

  // unsigned ImagePackH = H * W;
  // if (ImagePackH % BLOCK_MR != 0)
  //   ImagePackH += BLOCK_MR - ImagePackH % BLOCK_MR;

  // unsigned FilterPackW = FH * FW;
  // if (FilterPackW % BLOCK_NR != 0)
  //   FilterPackW += BLOCK_NR - FilterPackW % BLOCK_NR;

