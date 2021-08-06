#include "utils.h"
#include <stdlib.h>
#include <algorithm>
#include <blis.h>
#include <chrono>
#include <iostream>
#include <string>


using namespace std;


#define BLOCK_MR 6
#define BLOCK_NR 16
#define BLOCK_MC 168
#define BLOCK_KC 256
#define BLOCK_NC 4080


#define is_a_ge_zero_and_a_lt_b(a, b) ((a >= 0) && (a < b))
void im2col_cpu(const float *data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                float *data_col) {

  unsigned i = 0;

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
                // if (2 == data_im[input_row * width + input_col])
                // cout << input_row << " * " << width << " + " << input_col <<
                // "\n";
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
#undef is_a_ge_zero_and_a_lt_b

// Input: H x W
// L: OW x H x KW
float *mec(const float *Input, unsigned C, unsigned H, unsigned W, unsigned KH,
           unsigned KW, unsigned PadH, unsigned PadW, unsigned StrideH,
           unsigned StrideW, unsigned DilH, unsigned DilW) {

  const unsigned OH = (H - KH + 2 * PadH) / StrideH + 1;
  const unsigned OW = (W - KW + 2 * PadW) / StrideW + 1;

  auto *L = new float[OW * H * KW];

  for (unsigned w = 0; w < OW; ++w) {
    for (unsigned h = 0; h < H; ++h) {
      unsigned LIdx = (w * H + h) * KW;
      unsigned IIdx = h * W + w * StrideW;
      // L[w, h, 0:kw] = I[h, w * StrideW: w * StrideW + KW];
      for (unsigned kw = 0; kw < KW; ++kw)
        L[LIdx + kw] = Input[IIdx + kw];
    }
  }

  return L;
}

void conv_im2col(const float *Input, float *Kernel, float *Output, unsigned C,
                 unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
                 unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
                 unsigned StrideH, unsigned StrideW, unsigned DilH,
                 unsigned DilW) {

  // im2col
  float *BufIm2col = allocateTensor(C * KH * KW * OH * OW);
  im2col_cpu(Input, C, H, W, KH, KW, 0, 0, 1, 1, 1, 1, BufIm2col);
  cout << "=== BufIm2col ===\n";
  printTensor(BufIm2col, {C * KH * KW, OH * OW});
  cout << string(80, '-') << "\n\n";

  // Post-im2col GEMM
  {
    float alpha = 1.0, beta = 0.0;
    unsigned Mgemm = M;
    unsigned N = OH * OW;
    unsigned K = C * KH * KW;
    unsigned rsa = C * KW * KW;
    unsigned csa = 1;
    unsigned rsb = OH * OW;
    unsigned csb = 1;
    unsigned rsc = OH * OW;
    unsigned csc = 1;
    cout << Mgemm << " x " << K << " x " << N << " gemm\n";
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, Mgemm, N, K, &alpha, Kernel,
              rsa, csa, BufIm2col, rsb, csb, &beta, Output, rsc, csc);
    cout << "=== OutputIm2col ===\n";
    printTensor(Output, {Mgemm, N});
    cout << string(80, '-') << "\n\n";
  }
}

void conv_mec(const float *Input, float *Kernel, float *Output, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
              unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
              unsigned StrideH, unsigned StrideW, unsigned DilH,
              unsigned DilW) {

  // MEC
  float *BufMec = mec(Input, C, H, W, KH, KW, 0, 0, 1, 1, 1, 1);
  cout << "=== BufMec ===\n";
  printTensor(BufMec, {OW, H * KW});
  cout << string(80, '-') << "\n\n";

  // Post-mec GEMM
  {
    float alpha = 1.0, beta = 0.0;
    unsigned Mgemm = M;
    unsigned N = OW;
    unsigned K = C * KH * KW;
    unsigned rsa = C * KH * KW;
    unsigned csa = 1;
    unsigned rsb = H * KW;
    unsigned csb = 1;
    unsigned rsc = OH * OW;
    unsigned csc = 1;
    cout << Mgemm << " x " << K << " x " << N << " gemm\n";
    for (unsigned h = 0; h < OH; ++h) {
      bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, Mgemm, N, K, &alpha, Kernel,
                rsa, csa, BufMec + h * KW, rsb, csb, &beta, Output + h * OW,
                rsc, csc);
      // printTensor(tmp, {M, OW});
    }
    cout << "=== OutputMec ===\n";
    printTensor(Output, {M, OH * OW});
    cout << string(80, '-') << "\n\n";
  }
}

void packA(float *Block, float *&Pack, unsigned LDA, unsigned MC, unsigned KC) {
  unsigned MR = BLOCK_MR;

  for (unsigned ic = 0; ic < MC; ic += MR)
    for (unsigned k = 0; k < KC; ++k) {
      for (unsigned ir = 0; ir < MR; ++ir) {
        unsigned From = (ic + ir) * LDA + k;
        unsigned To = ic * KC + k * MR + ir;
        Pack[To] = Block[From];
    }
  }
}

void packAUnrolled(float *Block, float *&Pack, unsigned LDA, unsigned MC, unsigned KC) {
  unsigned MR = BLOCK_MR;

  for (unsigned k = 0; k < KC; ++k) {
    for (unsigned ic = 0; ic < MC; ic += MR) {
      float *BlockOff = Block + ic * LDA + k;
      float *PackOff = Pack + ic * KC + k * MR;
      // MR times (6)
      *(PackOff++) = *(BlockOff) + 0 * LDA;
      *(PackOff++) = *(BlockOff) + 1 * LDA;
      *(PackOff++) = *(BlockOff) + 2 * LDA;
      *(PackOff++) = *(BlockOff) + 3 * LDA;
      *(PackOff++) = *(BlockOff) + 4 * LDA;
      *(PackOff++) = *(BlockOff) + 5 * LDA;
    }
  }
}

void packB(float *Block, float *&Pack, unsigned LDB, unsigned KC, unsigned NC) {
  unsigned NR = BLOCK_NR;

  for (unsigned jc = 0; jc < NC; jc += NR)
    for (unsigned k = 0; k < KC; ++k) {
      for (unsigned jr = 0; jr < NR; ++jr) {
        unsigned From = jc + jr + LDB * k;
        unsigned To = jc * KC + k * NR + jr;
        Pack[To] = Block[From];
    }
  }
}

void packBUnrolled(float *Block, float *&Pack, unsigned LDB, unsigned KC, unsigned NC) {
  unsigned NR = BLOCK_NR;

  for (unsigned k = 0; k < KC; ++k) {
    for (unsigned jc = 0; jc < NC; jc += NR) {
      float *BlockOff = Block + jc + LDB * k;
      float *PackOff = Pack + jc * KC + k * NR;
      // NR times (16)
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
      *(PackOff++) = *(BlockOff++);
    }
  }
}

void mm(float *A, float *B, float *C, unsigned M, unsigned K, unsigned N,
        unsigned LDA, unsigned LDB, unsigned LDC) {

  unsigned MC = BLOCK_MC, KC = BLOCK_KC, NC = BLOCK_NC;
  unsigned MR = BLOCK_MR, NR = BLOCK_NR;

  float *APack = (float *)aligned_alloc(4096, MC * KC * sizeof(float));
  float *BPack = (float *)aligned_alloc(4096, KC * NC * sizeof(float));

  auto *cntx = bli_gks_query_cntx();
  auto *data = new auxinfo_t;

  float Alpha = 1.0, Beta = 0.0;
  for (unsigned jc = 0; jc < N; jc += NC)
    for (unsigned k = 0; k < K; k += KC) {
      Beta = k == 0 ? 0.0 : 1.0; // Accumulate
      packB(B + k * LDB + jc, BPack, LDB, KC, NC);
      for (unsigned ic = 0; ic < M; ic += MC) {
        packA(A + ic * LDA + k, APack, LDA, MC, KC);
        for (unsigned jr = 0; jr < NC; jr += NR)
          for (unsigned ir = 0; ir < MC; ir += MR) {
            bli_sgemm_haswell_asm_6x16(KC, &Alpha,
                APack + ir * KC, BPack + jr * KC, &Beta,
                C + (ic + ir) * LDC + jc + jr, LDC, 1,
                data, cntx);
          }
      }
    }
}

// void convGemm(float *Input, float *Kernel, float *Output, unsigned C,
//               unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW) {
//               //PackFuncT *packA, PackFuncT *packB) {
// 
//   // OH, OW
//   unsigned OH = H - KH + 1, OW = W - KW + 1;
// 
//   // gemm m x k x n
//   unsigned m = M, k = C * KH * KW, n = M * OH * OW;
// 
//   for (unsigned jc = 0; jc < n; jc += BLOCK_NC)
//     for (unsigned pc = 0; pc < k; pc += BLOCK_KC) {
//       for (unsigned ic = 0; ic < m; ic += BLOCK_MC) {
//       }
//     }
// }
