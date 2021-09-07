#pragma once
#include <vector>

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            float *data_col);

void convIm2col(const float *Input, float *Kernel, float *Output, unsigned C,
                unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
                unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
                unsigned StrideH, unsigned StrideW, unsigned DilH,
                unsigned DilW);

void packKernel(const float *Kernel, float *Pack, unsigned M, unsigned C, unsigned KH, unsigned KW);

void packInput(const float *Input, float *Pack, unsigned C, unsigned H, unsigned W, unsigned KH, unsigned KW, unsigned OW);

float *yaconv(const float *Input, float *Kernel, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
              unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
              unsigned StrideH, unsigned StrideW, unsigned DilH, unsigned DilW);

void mecNCHWTransformKernel(const float *Kernel, float *Output, unsigned M,
                            unsigned C, unsigned KH, unsigned KW);

void mecNCHWTransformInput(const float *Input, float *Output, unsigned C,
                           unsigned H, unsigned W, unsigned KH, unsigned KW);

float *convMecNCHW(const float *Input, const float *Kernel, unsigned C,
                   unsigned H, unsigned W, unsigned M, unsigned KH,
                   unsigned KW);

float *convGemm(const float *Input, const float *Kernel, unsigned C, unsigned H,
                unsigned W, unsigned M, unsigned KH, unsigned KW);
