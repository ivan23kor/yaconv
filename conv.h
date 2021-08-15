#ifndef _CONV_H_
#define _CONV_H_

void im2col_cpu(const float *data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                float *data_col);

float *mec(const float *Input, unsigned C, unsigned H, unsigned W, unsigned KH,
           unsigned KW, unsigned PadH, unsigned PadW, unsigned StrideH,
           unsigned StrideW, unsigned DilH, unsigned DilW);

void conv_im2col(const float *Input, float *Kernel, float *Output, unsigned C,
                 unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
                 unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
                 unsigned StrideH, unsigned StrideW, unsigned DilH,
                 unsigned DilW);

void conv_mec(const float *Input, float *Kernel, float *Output, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
              unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
              unsigned StrideH, unsigned StrideW, unsigned DilH, unsigned DilW);

void mm(float *A, float *B, float *C, unsigned M, unsigned K, unsigned N,
        unsigned LDA, unsigned LDB, unsigned LDC);

typedef float *(*PackFuncT)(float *);

void packB(float *Block, float *&Pack, unsigned LDB, unsigned KC, unsigned NC);

void packInputAsB(float *Input, float *&Pack, unsigned Ks, unsigned Ns, unsigned KC, unsigned NC, unsigned C, unsigned H, unsigned W, unsigned KH, unsigned KW, unsigned OW);

void convGemm(float *Input, float *Kernel, float *Output, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW);

#endif // _CONV_H_
