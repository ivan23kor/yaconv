#include <blis.h>
namespace {
#include "blis_params.hpp" // BLIS block sizes, microkernels
};
#include "utils.hpp"  // Tensor aligned allocation and printing
#include <chrono>     // Timing
#include <cmath>      // std::ceil
#include <iostream>   // Debug printing


#ifdef DEBUG_OUTPUT
#undef DEBUG_OUTPUT
#define DEBUG_OUTPUT(f) {f;}
#else
#define DEBUG_OUTPUT(f)
#endif

#ifdef DEBUG_TIME
#undef DEBUG_TIME
#define DEBUG_TIME(f) {f;}
#else
#define DEBUG_TIME(f)
#endif


// Timing
using namespace std::chrono;
static high_resolution_clock::time_point t1, t2, t3, t4;
double YaConvPack1 = 0.0, YaConvPack2 = 0.0, YaConvComp = 0.0;

// TODO: Convolution parameters as a struct
static int C, H, W, M, FH, FW, OH, OW, SH, SW, PH, PW;
static int YACONV_NC;

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

static float *ImagePack = nullptr;
static float *FilterPack = nullptr;
static float *CBuff = nullptr;

static void allocateYaconvBuffers() {
  if (ImagePack != nullptr)
    return;
  ImagePack = alignedAlloc(W * C * YACONV_NC);
  FilterPack = alignedAlloc(BLOCK_MC * BLOCK_KC);
  CBuff = alignedAlloc(BLOCK_MR * BLOCK_NR, BLIS_SIMD_ALIGN_SIZE);
}

void yaconv(float *Image, float *Filter, float *Output,
            int C_, int H_, int W_, int M_, int FH_, int FW_,
            int PH_, int PW_, int SH_, int SW_) {

  // Set convolution parameters to global values
  C = C_, H = H_, W = W_, M = M_, FH = FH_, FW = FW_, PH = PH_, PW = PW_, SH = SH_, SW = SW_;

  // Output sizes
  OH = (H - FH + 2 * PH) / SH + 1;
  OW = (W - FW + 2 * PW) / SW + 1;

  // Compute runtime-based slice of the image that fits in L3
  YACONV_NC = BLOCK_MC * BLOCK_NC / W / C;
  if (YACONV_NC % BLOCK_NR != 0)
    YACONV_NC += BLOCK_NR - YACONV_NC % BLOCK_NR;

  allocateYaconvBuffers();

  DEBUG_OUTPUT(
    printTensor(Image, {H, W * C});
    printTensor(Filter, {M, FH * FW * C});
  )

  for (int nc = 0; nc < H; nc += YACONV_NC) {
    int NC = MIN(H - nc, YACONV_NC);
    DEBUG_TIME(t1 = high_resolution_clock::now();)
    packImage(Image + nc * W * C, ImagePack, NC);
    DEBUG_TIME(
      t2 = high_resolution_clock::now();
      YaConvPack1 += duration_cast<duration<double>>(t2 - t1).count();
      t3 = high_resolution_clock::now();
    )
    DEBUG_OUTPUT(
      printTensor(ImagePack, {W * C * (int)std::ceil((float)H / BLOCK_NR), BLOCK_NR});
    )
    for (int fh = 0; fh < FH; ++fh) {
      for (int m = 0; m < M; m += BLOCK_MC) {
        int MC = MIN(M - m, BLOCK_MC);
        for (int kc = 0; kc < FW * C; kc += BLOCK_KC) {
          int KC = MIN(FW * C - kc, BLOCK_KC);
          DEBUG_TIME(t1 = high_resolution_clock::now();)
          packFilter(Filter + (m * FH + fh) * FW * C + kc, FilterPack, MC, KC);
          DEBUG_TIME(
            t2 = high_resolution_clock::now();
            YaConvPack2 += duration_cast<duration<double>>(t2 - t1).count();
          )
          DEBUG_OUTPUT(printTensor(FilterPack, {KC * (int)std::ceil((float)MC / BLOCK_MR), BLOCK_MR});)
          for (int nr = 0; nr < NC; nr += BLOCK_NR) {
            for (int ow = 0; ow < OW; ++ow) {
              int ImageStart = (ow - PW) * C + kc;
              int ImageEnd = MIN(W * C, ImageStart + KC);

              float *Ar = FilterPack;
              if (ImageStart < 0) {
                Ar -= ImageStart * BLOCK_MR;
                ImageStart = 0;
              }

              int K = ImageEnd - ImageStart;
              if (K <= 0)
                continue;

              float *Br = ImagePack + nr * W * C + ImageStart * BLOCK_NR;
              float *Cr = Output + ((nc + nr - fh + PH) * OW + ow) * M + m;

              for (int mr = 0; mr < MC; mr += BLOCK_MR) {
                DEBUG_OUTPUT(
                  std::cout << "Going to gemm Weight:\n";
                  printTensor(Ar, {K, BLOCK_MR});
                  std::cout << "and Image" << std::endl;
                  printTensor(Br, {K, BLOCK_NR});
                )
                // bli_sset0s_mxn(1, BLOCK_MR, Cr, OW * M, 1);
                if (mr + BLOCK_MR <= MC) {
                  sgemm_ukr(K, bli_s1, Ar, Br, bli_s1, Cr, 1, OW * M);
                } else {
                  sgemm_ukr(K, bli_s1, Ar, Br, bli_s0, CBuff, BLOCK_NR, 1);
                  bli_sxpbys_mxn(MC - mr, BLOCK_NR, CBuff, BLOCK_NR, 1, bli_s1, Cr, 1, OW * M);
                }
                DEBUG_OUTPUT(
                  std::cout << "so now the result is:" << std::endl;
                  printTensor(Output, {OH * OW, M});
                )

                Ar += BLOCK_MR * KC;
                Cr += BLOCK_MR;
              }
            }
          }
        }
      }
    }
  }

  DEBUG_TIME(
    t4 = high_resolution_clock::now();
    YaConvComp += duration_cast<duration<double>>(t4 - t3).count();
  )
  // TODO: deallocate buffers
}
