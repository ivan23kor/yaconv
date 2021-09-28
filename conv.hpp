#pragma once

#include <vector>

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            float *data_col);

void convIm2col(const float *Input, float *Kernel, float *Output, unsigned C,
                unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
                unsigned OH, unsigned OW, unsigned PH, unsigned PW,
                unsigned SH, unsigned SW);

float *convGemm(const float *Input, const float *Kernel, unsigned C, unsigned H,
                unsigned W, unsigned M, unsigned KH, unsigned KW);
