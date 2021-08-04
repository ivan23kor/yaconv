#include "blis.h"
#include "utils.h"
#include <iostream>
#include <string>

using namespace std;

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
