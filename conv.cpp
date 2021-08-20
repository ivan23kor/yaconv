#include "conv.h"
#include "gemm.h"
#include "utils.h"
#include <blis.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>


using namespace std;
using namespace std::chrono;



#define CONV_DEBUG(expr) if (DEBUG == 1) {expr;}
#define MIN(a, b) a < b ? a : b


namespace {

#include "set_blis_params.h"

}; // abstract namespace

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Im2col +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// The following two function are taken from https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
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

  // im2col
  float *BufIm2col = allocateTensor(C * KH * KW * OH * OW);
  im2col(Input, C, H, W, KH, KW, 0, 0, 1, 1, 1, 1, BufIm2col);
  cout << "=== BufIm2col (" << C * KH * KW * OH * OW << " elements) ===\n";
  CONV_DEBUG(
    printTensor(BufIm2col, {C * KH * KW, OH * OW});
    cout << string(80, '-') << "\n\n";
  )

  // Post-im2col GEMM
  {
    float alpha = 1.0, beta = 0.0;
    unsigned N = OH * OW;
    unsigned K = C * KH * KW;
    unsigned rsa = K;
    unsigned csa = 1;
    unsigned rsb = N;
    unsigned csb = 1;
    unsigned rsc = N;
    unsigned csc = 1;
    CONV_DEBUG(cout << M << " x " << K << " x " << N << " gemm\n";)
    gemm(Kernel, BufIm2col, Output, M, K, N, K, N, N, 1.0, 0.0);
    // bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &alpha, Kernel,
    //           rsa, csa, BufIm2col, rsb, csb, &beta, Output, rsc, csc);
  }
}
//---------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Fused im2col & packing +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Input: C x H x W
inline void im2colPackB(float *Input, float *&Pack, unsigned Ks, unsigned Ns, unsigned KC, unsigned NC, unsigned C, unsigned H, unsigned W, unsigned KH, unsigned KW, unsigned OW) {
  unsigned To = 0;
  for (unsigned jc = 0; jc < NC; jc += BLOCK_NR) {
    for (unsigned k = 0; k < KC; ++k) {
      unsigned NR = MIN(NC - jc, BLOCK_NR);
      for (unsigned jr = 0; jr < NR; ++jr) {
        unsigned patch = Ns + jc + jr;
        unsigned channel = (Ks + k) / (KH * KW);
        unsigned patch_row = ((Ks + k) % (KH * KW)) / KW;
        unsigned patch_col = ((Ks + k) % (KH * KW)) % KW;
        unsigned From = channel * H * W + (patch / OW) * W + patch % OW + patch_row * W + patch_col;
        Pack[To++] = Input[From];
      }
      for (unsigned End = To + BLOCK_NR - NR; To != End; ++To)
        Pack[To] = 0.0;
    }
  }
}

void convGemm(float *Input, float *Kernel, float *Output, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW) {

  auto *APack = (float *)aligned_alloc(4096, BLOCK_MC * BLOCK_KC * sizeof(float));
  auto *BPack = (float *)aligned_alloc(4096, BLOCK_KC * BLOCK_NC * sizeof(float));
  auto *CBuff = (float *)aligned_alloc(4096, BLOCK_MR * BLOCK_NR * sizeof(float));

  const unsigned OH = H - KH + 1, OW = W - KW + 1;

  const unsigned K = C * KH * KW;
  const unsigned N = OH * OW;
  const unsigned LDA = K, LDC = N;

  float Alpha = 1.0, Zero = 0.0;
  high_resolution_clock::time_point t1, t2;
  double PackATime = 0.0, PackBTime = 0.0;
  for (unsigned jc = 0; jc < N; jc += BLOCK_NC) {

    unsigned NC = MIN(N - jc, BLOCK_NC);

    for (unsigned k = 0; k < K; k += BLOCK_KC) {

      unsigned KC = MIN(K - k, BLOCK_KC);

      float Beta = k == 0 ? 0.0 : 1.0; // Accumulate or not

      t1 = high_resolution_clock::now();
      im2colPackB(Input, BPack, k, jc, KC, NC, C, H, W, KH, KW, OW);
      t2 = high_resolution_clock::now();
      PackBTime += duration_cast<duration<double>>(t2 - t1).count();

      for (unsigned ic = 0; ic < M; ic += BLOCK_MC) {

        unsigned MC = MIN(M - ic, BLOCK_MC);

        t1 = high_resolution_clock::now();
        packA(Kernel + ic * LDA + k, APack, LDA, MC, KC);
        t2 = high_resolution_clock::now();
        PackATime += duration_cast<duration<double>>(t2 - t1).count();

        for (unsigned jr = 0; jr < NC; jr += BLOCK_NR) {

          unsigned NR = MIN(NC - jr, BLOCK_NR);

          for (unsigned ir = 0; ir < MC; ir += BLOCK_MR) {

            unsigned MR = MIN(MC - ir, BLOCK_MR);

            float *Ar = APack + ir * KC;
            float *Br = BPack + jr * KC;
            float *Cr = Output + (ic + ir) * LDC + jc + jr;

            if ((MR == BLOCK_MR) && (NR == BLOCK_NR))
              blisGemmUKR(KC, &Alpha, Ar, Br, &Beta, Cr, LDC, 1, data, cntx);
            else {
              blisGemmUKR(KC, &Alpha, Ar, Br, &Zero, CBuff, BLOCK_NR, 1,
                  data, cntx);
              bli_saxpym(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
                  MR, NR, &Alpha, CBuff, BLOCK_NR, 1, Cr, LDC, 1);
            }
          }
        }
      }
    }
  }
  std::cout << "PackATime: " << PackATime << "\n";
  std::cout << "PackBTime: " << PackBTime << "\n";
}
//---------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Yaconv +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Input: C x H x W
void yaconvPackB(float *Input, float *&Pack, unsigned Ks, unsigned Ns, unsigned KC, unsigned NC, unsigned C, unsigned H, unsigned W, unsigned KH, unsigned KW, unsigned OW) {
  unsigned To = 0;
  for (unsigned jc = Ns; jc < Ns + NC; jc += BLOCK_NR) {
    for (unsigned k = Ks; k < Ks + KC; ++k) {
      unsigned NR = MIN(NC - jc, BLOCK_NR);
      for (unsigned jr = 0; jr < NR; ++jr) {
        unsigned channel = k / (H * KW);
        unsigned row = (k % (H * KW)) / KW;
        unsigned col = (k % (H * KW)) % KW;
        unsigned From = channel * H * W + row * W + col + jc + jr;
        Pack[To++] = Input[From];
      }
      for (unsigned End = To + BLOCK_NR - NR; To != End; ++To)
        Pack[To] = 0.0;
    }
  }
}
//---------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ MEC +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Input: H x W
// Output: OW x H x KW
void mec(const float *Input, unsigned C, unsigned H, unsigned W, unsigned KH,
         unsigned KW, unsigned PadH, unsigned PadW, unsigned StrideH,
         unsigned StrideW, unsigned DilH, unsigned DilW, float *Output) {

  const unsigned OH = (H - KH + 2 * PadH) / StrideH + 1;
  const unsigned OW = (W - KW + 2 * PadW) / StrideW + 1;

  for (unsigned ow = 0; ow < OW; ++ow) {
    for (unsigned c = 0; c < C; ++c) {
      for (unsigned h = 0; h < H; ++h) {
        unsigned OutputIdx = (c * H + h) * KW * OW + ow;
        unsigned InputIdx = c * H * W + h * W + ow;
        for (unsigned kw = 0; kw < KW; ++kw)
          Output[OutputIdx + kw * OW] = Input[InputIdx + kw];
      }
    }
  }
}

void convMEC(const float *Input, float *Kernel, float *Output, unsigned C,
             unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
             unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
             unsigned StrideH, unsigned StrideW, unsigned DilH,
             unsigned DilW) {

  // MEC
  float *BufMEC = allocateTensor(C * W * KH * OW);
  mec(Input, C, H, W, KH, KW, 0, 0, 1, 1, 1, 1, BufMEC);
  CONV_DEBUG(
    cout << "=== BufMec ===\n";
    printTensor(BufMEC, {C * H * KW, OW});
    cout << string(80, '-') << "\n\n";
  )

  // Post-mec GEMM
  {
    float alpha = 1.0, beta = 0.0;
    unsigned N = OH * OW;
    unsigned K = C * KH * KW;
    unsigned rsa = C * KH * KW;
    unsigned csa = 1;
    unsigned rsb = C * H * KW;
    unsigned csb = 1;
    unsigned rsc = OH * OW;
    unsigned csc = 1;
    CONV_DEBUG(cout << M << " x " << K << " x " << N << " gemm\n";)
    for (unsigned h = 0; h < OH; ++h) {
      bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, M, N, K, &alpha, Kernel,
                rsa, csa, BufMEC + h * KW, rsb, csb, &beta, Output + h * OW,
                rsc, csc);
    }
  }
}
//---------------------------------------------------------------------------
