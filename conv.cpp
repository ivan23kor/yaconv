#include "gemm.hpp"   // custom gemm
#include "utils.hpp"  // Tensor aligned allocation and printing
#include <iostream>   // debug printing

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

void convIm2col(const float *Input, float *Kernel, float *Output, int C,
                int H, int W, int M, int KH, int KW,
                int OH, int OW, int PH, int PW,
                int SH, int SW) {

  float *InputBuf = alignedAlloc(C * KH * KW * OH * OW);
  im2col(Input, C, H, W, KH, KW, PH, PW, SH, SW, 1, 1, InputBuf);

  int K = C * KH * KW;
  int N = OH * OW;

  // std::cout << "im2col buffer: " << C * KH * KW << " x " << OH * OW << "\n";
  // std::cout << "im2col GEMM: " << M << " x " << K << " x " << N << "\n";

  gemm(Kernel, InputBuf, Output, M, K, N, K, N, N, 1.0, 0.0);

  // int rsa = K;
  // int csa = 1;
  // int rsb = N;
  // int csb = 1;
  // int rsc = N;
  // int csc = 1;
  // float alpha = 1.0, beta = 0.0;
  // bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &alpha, Kernel,
  //           rsa, csa, InputBuf, rsb, csb, &beta, Output, rsc, csc);

  delete[] InputBuf;
}
