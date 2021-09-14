#include "gemm.hpp"
#include "utils.h"
#include <blis.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>

using namespace std::chrono;

namespace {
#include "set_blis_params.h"

high_resolution_clock::time_point t1, t2;
}; // namespace

#define CONV_DEBUG(expr)                                                       \
  if (DEBUG == 1) {                                                            \
    expr;                                                                      \
  }

#define MIN(a, b) a < b ? a : b

// A dirty macro to time parts of the algorithms
#ifndef TIME
#define TIME(cmd)                                                              \
  t1 = high_resolution_clock::now();                                           \
  cmd;                                                                         \
  t2 = high_resolution_clock::now();                                           \
  Times.push_back(duration_cast<duration<double>>(t2 - t1).count());
#else
#undef TIME
#define TIME(cmd) cmd;
#endif

extern std::vector<double> Times;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Im2col +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// The following two functions are taken from
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            float *data_col) {

  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void convIm2col(const float *Input, float *Kernel, float *Output, unsigned C,
                unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
                unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
                unsigned StrideH, unsigned StrideW, unsigned DilH,
                unsigned DilW) {

  TIME(float *InputBuf = alignedAlloc(C * KH * KW * OH * OW);)
  TIME(im2col(Input, C, H, W, KH, KW, 0, 0, 1, 1, 1, 1, InputBuf);)

  unsigned K = C * KH * KW;
  unsigned N = OH * OW;
  unsigned rsa = K;
  unsigned csa = 1;
  unsigned rsb = N;
  unsigned csb = 1;
  unsigned rsc = N;
  unsigned csc = 1;

  CONV_DEBUG(std::cout << "=== Im2col ===\n";
             std::cout << "InputBuf: " << C * KH * KW << " x " << OH * OW
                       << "\n";
             std::cout << "1 x GEMM[" << M << " x " << K << " x " << N << "]\n";
             std::cout << std::string(80, '-') << "\n\n";)

  TIME(gemm(Kernel, InputBuf, Output, M, K, N, K, N, N, 1.0, 0.0);)
  // TODO: For a fair comparison, calling custom gemm that is similar to
  // other convolution implementations in this file. Replace with a call to
  // OpenBLAS/BLIS/MKL/ESSL later.
  // bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &alpha, Kernel,
  //           rsa, csa, InputBuf, rsb, csb, &beta, Output, rsc, csc);

  delete[] InputBuf;
}
//---------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ MEC for NCHW input +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// Kernel: C x KH x KW
// Output: KH x KW x C
void mecNCHWTransformKernel(const float *Kernel, float *Output, unsigned M,
                            unsigned C, unsigned KH, unsigned KW) {
  unsigned To = 0;

  for (unsigned m = 0; m < M; ++m) {
    for (unsigned row = 0; row < KH; ++row) {
      for (unsigned col = 0; col < KW; ++col) {
        for (unsigned channel = 0; channel < C; ++channel) {
          unsigned From = m * C * KH * KW + channel * KH * KW + row * KW + col;
          Output[To++] = Kernel[From];
        }
      }
    }
  }
}

// Input: C x H x W
// Output: C * H * KW x OW, NHWC-MEC
void mecNCHWTransformInput(const float *Input, float *Output, unsigned C,
                           unsigned H, unsigned W, unsigned KH, unsigned KW) {
  unsigned OH = H - KH + 1;
  unsigned OW = W - KW + 1;

  unsigned To = 0;

  for (unsigned row = 0; row < H; ++row) {
    for (unsigned col = 0; col < KW; ++col) {
      for (unsigned channel = 0; channel < C; ++channel) {
        for (unsigned patchCol = 0; patchCol < OW; ++patchCol) {
          unsigned From = channel * H * W + row * W + col + patchCol;
          Output[To++] = Input[From];
        }
      }
    }
  }
}

auto *convMecNCHW(const float *Input, const float *Kernel, unsigned C,
                  unsigned H, unsigned W, unsigned M, unsigned KH,
                  unsigned KW) {

  unsigned OH = H - KH + 1, OW = W - KW + 1;

  TIME(
  auto *Output = alignedAlloc(M * OH * OW);
  auto *KernelBuf = alignedAlloc(M * KH * KW * C);
  auto *InputBuf = alignedAlloc(C * H * KW * OW);
  )

  TIME(mecNCHWTransformKernel(Kernel, KernelBuf, M, C, KH, KW);)
  TIME(mecNCHWTransformInput(Input, InputBuf, C, H, W, KH, KW);)

  unsigned K = KH * KW * C;
  unsigned N = OW;
  unsigned LDA = K, LDB = N, LDC = OH * N;

  TIME(for (unsigned oh = 0; oh < OH; ++oh)
           gemm(KernelBuf, InputBuf + oh * C * KW * OW, Output + oh * OW, M, K,
                N, LDA, LDB, LDC, 1.0, 0.0);)

  CONV_DEBUG(std::cout << "=== Mec-NCHW ===\n";
             std::cout << "KernelBuf: " << M << " x " << KH * KW * C << "\n";
             std::cout << "InputBuf: " << C * H * KW << " x " << OW << "\n";
             std::cout << OH << " x GEMM[" << M << " x " << K << " x " << N
                       << "]\n";
             std::cout << std::string(80, '-') << "\n\n";)

  //printTensor(InputBuf, {H * KW * C, OW});

  delete[] KernelBuf, InputBuf;

  return Output;
}
//---------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ YaConv +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void packKernel(const float *Kernel, float *Pack, unsigned M, unsigned C, unsigned KH, unsigned KW) {
  unsigned To = 0;
  for (unsigned ic = 0; ic < M; ic += BLOCK_MR) {
    unsigned MR = MIN(M - ic, BLOCK_MR);
    for (unsigned k = 0; k < KH * KW * C; ++k) {
      for (unsigned ir = 0; ir < BLOCK_MR; ++ir) {
        unsigned m = ic + ir;
        unsigned c = k % C;
        unsigned h = k / C / KW;
        unsigned w = k / C % KW;
        unsigned From = m * C * KH * KW + c * KH * KW + h * KW + w;
        // std::cout << From << " -> " << To++ << "\n";
        Pack[To++] = ir < MR ? Kernel[From] : 0.0;
      }
    }
  }
}

void packInput(const float *Input, float *Pack, unsigned C, unsigned H, unsigned W, unsigned KH, unsigned KW, unsigned OW) {
  unsigned To = 0;
  for (unsigned jc = 0; jc < OW; jc += BLOCK_NR) {
    unsigned NR = MIN(OW - jc, BLOCK_NR);
    for (unsigned k = 0; k < H * KW * C; ++k) {
      for (unsigned jr = 0; jr < BLOCK_NR; ++jr) {
        unsigned c = k % C;
        unsigned h = k / C / KW;
        unsigned w = k / C % KW;
        unsigned From = c * H * W + h * W + w + jc + jr;
        // std::cout << From << " -> " << To++ << "\n";
        Pack[To++] = jr < NR ? Input[From] : 0.0;
      }
    }
  }
}

float *yaconv(const float *Input, float *Kernel, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
              unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
              unsigned StrideH, unsigned StrideW, unsigned DilH, unsigned DilW) {

  unsigned BLOCK_OW = OW % BLOCK_NR == 0 ? OW : OW + BLOCK_NR - OW % BLOCK_NR;
  unsigned BLOCK_M = M % BLOCK_MR == 0 ? M : M + BLOCK_MR - M % BLOCK_MR;

  TIME(
  auto *Output = alignedAlloc(M * OH * OW);
  auto *KernelPack = alignedAlloc(BLOCK_M * KH * KW * C);
  auto *InputPack = alignedAlloc(H * KW * C * BLOCK_OW);
  )

  TIME(packKernel(Kernel, KernelPack, M, C, KH, KW);)
  TIME(packInput(Input, InputPack, C, H, W, KH, KW, OW);)

  CONV_DEBUG(
  std::cout << "Packed Kernel:\n";
  printTensor(KernelPack, {KH * KW * C * BLOCK_M / BLOCK_MR, BLOCK_MR});
  std::cout << "Packed Input:\n";
  printTensor(InputPack, {H * KW * C * BLOCK_OW / BLOCK_NR, BLOCK_NR});
  )

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
//---------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Fused im2col & packing +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// The following function is taken from
// https://gitlab.com/comtacts/convgemm/-/blob/public/convGemm.c#L1248-1336
void sPack_im2Col(unsigned int i, unsigned int j, const float *In,
                  float *B_pack, unsigned int k, unsigned int n, unsigned int c,
                  unsigned int h, unsigned int w, unsigned int ho,
                  unsigned int wo, unsigned int kh, unsigned int kw,
                  unsigned int hStride, unsigned int wStride) {
  unsigned int ic, ikw,
      ikh,                 // Row related indexes (regarding the phantom matrix)
      j_local, ib, iw, ih, // Col related indexes (regarding the phantom matrix)
      pos, pos_ic, pos_ib, pos_ic_ikw; // position on the original image
  unsigned int pos_ic_ini, ikw_ini, ikh_ini, pos_ib_ini, iw_ini,
      ih_ini; // Initial values of indexes

  unsigned int cSize = h * w, // chanel memory leap in input tensor
      coSize = ho * wo,       // chanel memory leap in matrix B
      kSize = kh * kw,        // kernel memory leap (single chanel)
      bSize = c * h * w;      // batch memory leap

  unsigned int jc, pc, jr; // loop control indexes
  float *restrict B_pack_local;
  unsigned int skipPos;

  ic = i / kSize;
  ikw_ini = (i % kSize) / kh;
  ikh_ini = (i % kSize) % kh;
  pos_ic_ini = ic * cSize;

  //#pragma omp parallel for private(B_pack_local,skipPos,
  // j_local,pc,jr,ib,ih_ini, iw_ini,
  // pos_ib_ini,pos_ic,ikw,pos_ic_ikw,ikh,pos_ib,iw,ih,pos) firstprivate(j)
  for (jc = 0; jc < n; jc += BLOCK_NR) {

    B_pack_local = &B_pack[jc * k];
    unsigned int n_alg = fmin(BLOCK_NR, n - jc);
    skipPos = BLOCK_NR - n_alg;

    j_local = j + jc;
    ib = j_local / coSize;
    iw_ini = (j_local % (coSize)) / ho;
    ih_ini = (j_local % (coSize)) % ho;
    pos_ib_ini = ib * bSize;

    // ih_ini = ih_ini + jc

    pos_ic = pos_ic_ini;
    ikw = ikw_ini;
    pos_ic_ikw = ikw * h + pos_ic;
    for (pc = 0, ikh = ikh_ini; pc < k; pc++, ikh++) {
      if (ikh == kh) {
        ikh = 0;
        ikw++;
        pos_ic_ikw += h; // OPT pos_ic_ikw = ikw* h +pos_ic
        if (ikw == kw) {
          ikw = 0;
          pos_ic += cSize;     // OPT ic++;pos_ic = ic * cSize;
          pos_ic_ikw = pos_ic; // OPT pos_ic_ikw = ikw *h + pos_ic;
        }
      }

      pos_ib = pos_ib_ini;
      iw = iw_ini;
      for (jr = 0, ih = ih_ini; jr < n_alg; jr++, ih++) {
        if (ih == ho) {
          ih = 0;
          iw++;
          if (iw == wo) {
            iw = 0;
            pos_ib += bSize; // OPT ib++;pos_in = ib*bSize;
          }
        }
        // OPT pos = ib * bSize  + ic * cSize + (iw * wStride + ikw) *h + (ih *
        // hStride + ikh); OPT pos = pos_ib + pos_ic + (iw * wStride * h +
        // pos_ikw) + (ih * hStride + ikh);
        pos = pos_ib + pos_ic_ikw + iw * wStride * h + (ih * hStride + ikh);

        B_pack_local[0] = In[pos];
        B_pack_local++;
      }
      B_pack_local += skipPos;
    }
    // ih_ini = ih;
    // iw_ini = iw;
    // pos_ib_ini = pos_ib;
  }
}

auto *convGemm(const float *Input, const float *Kernel, unsigned C_, unsigned H,
               unsigned W, unsigned M, unsigned KH, unsigned KW) {

  auto *APack = alignedAlloc(BLOCK_MC * BLOCK_KC);
  auto *BPack = alignedAlloc(BLOCK_KC * BLOCK_NC);
  auto *CBuff = alignedAlloc(BLOCK_MR * BLOCK_NR);

  unsigned OH = H - KH + 1, OW = W - KW + 1;

  // GEMM sizes
  // unsigned M = M is the number of output channels
  unsigned K = C_ * KH * KW;
  unsigned N = OH * OW;
  unsigned LDA = K, LDC = N;

  const float *A = Kernel, *B = Input;
  auto *C = alignedAlloc(M * OH * OW);

  float Alpha = 1.0, Beta = 0.0;

  // C *= Beta
  TIME(
  bli_sscalm(BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, M, N, &Beta, C, LDC, 1);
  )

  float Zero = 0.0, One = 1.0;
  for (unsigned jc = 0; jc < N; jc += BLOCK_NC) {

    unsigned NC = MIN(N - jc, BLOCK_NC);

    for (unsigned k = 0; k < K; k += BLOCK_KC) {

      unsigned KC = MIN(K - k, BLOCK_KC);

      sPack_im2Col(k, jc, Input, BPack, KC, NC, C_, H, W, OH, OW, KH, KW, 1, 1);

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
              blisGemmUKR(KC, &Alpha, Ar, Br, &One, Cr, LDC, 1, data, cntx);
            else {
              blisGemmUKR(KC, &Alpha, Ar, Br, &Zero, CBuff, BLOCK_NR, 1, data,
                          cntx);
              // TODO: scal2m can be replaced with copym for convolution.
              // Should be faster, keeping it like that to be fair with the
              // performance of gemm based on scal2m.
              bli_saxpym(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
                         MR, NR, &Alpha, CBuff, BLOCK_NR, 1, Cr, LDC, 1);
            }
          }
        }
      }
    }
  }
  return C;
}
//---------------------------------------------------------------------------
