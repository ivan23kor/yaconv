#include "gemm.hpp"  // Tensor aligned allocation and printing
#include "utils.hpp"  // Tensor aligned allocation and printing
#include "yaconv.hpp" // class Indexer
#include <blis.h>     // BLIS microkernel and block sizes
#include <chrono>     // Timing
#include <iostream>   // Debug printing
#include <set>        // Group construction

// Timing
using namespace std::chrono;
static high_resolution_clock::time_point t1, t2, t3, t4, t5, t6;
double PackImageTime = 0.0, PackFilterTime = 0.0;

namespace {
#include "blis_params.hpp"
}; // namespace

// Convolution parameters as global variables for this compilation unit.
// Saves a lot of time passing them around different functions.
// static int C, H, W, FH, FW, PH, PW, SH, SW;
// C = C_, H = H_, W = W_, FH = FH_, FW = FW_, PH = PH_, PW = PW_, SH = SH_, SW = SW_;

// Defined in test_conv.cpp (driver code for convolutions)
// Algorithms in this file will append times to this vector
extern std::vector<double> Times;

void packImage(const float *Image, float *Pack, int MS, int MC,
    int C, int H, int W) {

  int To = 0;
  for (int ic = MS; ic < MS + MC; ic += BLOCK_MR) {
    int MR = MIN(MS + MC - ic, BLOCK_MR);
    for (int c = 0; c < C; ++c) {
      for (int ir = 0; ir < MR; ++ir) {
        int From = c * H * W + ic + ir;
        Pack[To++] = Image[From];
      }
      To += BLOCK_MR - MR;
    }
  }
}

void packFilter(const float *Filter, float *Pack, int NS, int NC,
    int M, int C, int FH, int FW) {

  int To = 0;
  for (int jc = NS; jc < NS + NC; jc += BLOCK_NR) {
    int NR = MIN(NS + NC - jc, BLOCK_NR);
    for (int c = 0; c < C; ++c) {
      for (int jr = 0; jr < NR; ++jr) {
        int m = (jc + jr) % M;
        int fh = (jc + jr) / M / FW;
        int fw = (jc + jr) / M % FW;
        int From = c * FH * FW + m * C * FH * FW + fh * FW + fw;
        Pack[To++] = Filter[From];
      }
      To += BLOCK_NR - NR;
    }
  }
}

// Assumptions: stride == 1
void packImageCenter(float *ImageOff, float *Pack, int MS, int MC,
    int C, int H, int W, int CenterW, int SkipW) {

  for (int i = 0; i < MC; ++i) {
    int ImageIdx = i + i / CenterW * SkipW;
    int To = (i - i % BLOCK_MR) * C + i % BLOCK_MR;
    for (int c = 0, From = ImageIdx; c < C; ++c, From += H * W) {
      Pack[To++] = ImageOff[From];
    }
  }
}

// inline void copyBufferToOutput(float *TmpOutput, float *Output,
//     int ImageStart, int ImageEnd, int FilterStart, int FilterEnd) {
//   for (int j = FilterStart; j < FilterEnd; ++j) {
//     int m = j / FW % M;
//     int fh = j / FW / M;
//     int fw = j % FW;
//     for (int i = ImageStart; i < ImageEnd; ++i) {
//       int h = i
//     }
//   }
// }

void yaconv(float *Image, float *Filter, float *Output, int C,
            int H, int W, int M, int FH, int FW,
            int SH, int SW, int PH, int PW) {

  // TODO: remove this when non-unit stride becomes supported
  if ((SH != 1) || (SW != 1)) {
    std::cout << "\033[31mSorry, only accepting unit stride convolutions for the moment\033[0m\n";
    exit(-1);
  }

  // Output sizes
  const int OH = (H - FH + 2 * PH) / SH + 1;
  const int OW = (W - FW + 2 * PW) / SW + 1;

  // Center gaps, sizes
  int GapW = FW - 1 - PW, GapH = FH - 1 - PH;
  int CenterW = W - 2 * GapW, CenterH = H - 2 * GapH;
  int CenterSize = CenterH * CenterW;
  int Off = W * (FH - 1 - PH) + GapW;
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

  // GEMM alpha, beta
  float Alpha = 1.0, One = 1.0;

  // NC loop
  for (int nc = 0; nc < M * FH * FW; nc += BLOCK_NC) {

    int NC = MIN(M * FH * FW - nc, BLOCK_NC);

#define DO_NOT_TIME_PACK_FILTER_TIME 1
#ifndef DO_NOT_TIME_PACK_FILTER_TIME
    t3 = high_resolution_clock::now();
#endif
    packFilter(Filter, FilterPack, nc, NC, M, C, FH, FW);
#ifndef DO_NOT_TIME_PACK_FILTER_TIME
    t4 = high_resolution_clock::now();
    PackFilterTime += duration_cast<duration<double>>(t4 - t3).count();
#endif
    // printTensor(FilterPack, {C * (int)std::ceil((float)NC / (float)BLOCK_NR), BLOCK_NR});

    for (int mc = 0; mc < CenterSize; mc += BLOCK_MC) {

      int MC = MIN(CenterSize - mc, BLOCK_MC);

#define DO_NOT_TIME_PACK_IMAGE_TIME 1
#ifndef DO_NOT_TIME_PACK_IMAGE_TIME
      t5 = high_resolution_clock::now();
#endif
      packImageCenter(Image + Off, ImagePack, mc, MC, C, H, W, CenterW, 2 * GapW);
#ifndef DO_NOT_TIME_PACK_IMAGE_TIME
      t6 = high_resolution_clock::now();
      PackImageTime += duration_cast<duration<double>>(t6 - t5).count();
#endif
      IF_DEBUG(printTensor(ImagePack, {C * (int)std::ceil((float)MC / (float)BLOCK_MR), BLOCK_MR});)

      for (int nr = 0; nr < NC; nr += BLOCK_NR) {

        int NR = MIN(NC - nr, BLOCK_NR);

        for (int mr = 0; mr < MC; mr += BLOCK_MR) {

          int MR = MIN(MC - mr, BLOCK_MR);

          // std::cout << "======================================================\n";
          // printTensor(ImagePack + mr * C, {C, BLOCK_MR});
          // std::cout << "\n\t*\n\n";
          // printTensor(FilterPack + nr * C, {C, BLOCK_NR});
          // std::cout << "\n\t=\n\n";

          // Microkernel call
          blisGemmUKR(C, &Alpha, ImagePack + mr * C, FilterPack + nr * C, &One, TmpOutput, BLOCK_NR, 1, data, cntx);
          // copyBufferToOutput(TmpOutput, Output, mc + mr, nc + nr, MR, NR);

          // Copy to correct output location
          // int fh = (nc + nr) / M / FW, fw = (nc + nr) / M % FW;
          // std::cout << "\033[32mfh = " << fh << ", fw = " << fw << "\033[0m\n";

          // printTensor(TmpOutput, {BLOCK_MR, BLOCK_NR});
          // std::cout << "\n\t-\n\n";
          // std::cout << "There are " << BLOCK_MR << " x " << BLOCK_NR << " elements.\n";
          // for (int row = mc + mr; row < mc + mr + MR; ++row) {
          //   std::cout << "Row[" << row << "] \n";
          //   for (int col = nc + nr; col < nc + nr + NR; ++col) {
          //     int m = col % M;
          //     int fh = col / M / FW;
          //     int fw = col / M % FW;
          //     int pos = W * (FH - 1 - PH) + GapW + row / CenterW ;
          //     std::cout << col << "(" << m << ", " << fh << ", " << fw << ", " << pos << ") ";
          //   }
          //   std::cout << "\n";
          // }
          // std::cout << "======================================================\n\n";
        }
      }
    }
  }
}

  // // Compute groups
  // for (int h = 0; h < H; ++h) {
  //   for (int w = 0; w < W; ++w) {
  //     int ImageIdx = h * W + w;
  //     std::set<int> FilterSet;
  //     for (int kh = 0; kh < FH; ++kh) {
  //       for (int kw = 0; kw < FW; ++kw) {
  //         int FilterIdx = kh * FW + kw;
  //         if    ((h + PH >= kh) && (h + (FH - 1 - kh) < H + PH)   // height within padded area
  //             && (w + PW >= kw) && (w + (FW - 1 - kw) < W + PW))  // width within padded area
  //           FilterSet.insert(FilterIdx);
  //       }
  //     }
  //   }
  // }

  // int ImagePackH = H * W;
  // if (ImagePackH % BLOCK_MR != 0)
  //   ImagePackH += BLOCK_MR - ImagePackH % BLOCK_MR;

  // int FilterPackW = FH * FW;
  // if (FilterPackW % BLOCK_NR != 0)
  //   FilterPackW += BLOCK_NR - FilterPackW % BLOCK_NR;

