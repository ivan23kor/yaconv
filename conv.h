#ifndef _CONV_H_
#define _CONV_H_

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ im2col +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
//-------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ MEC +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void mec(const float *Input, unsigned C, unsigned H, unsigned W, unsigned KH,
         unsigned KW, unsigned PadH, unsigned PadW, unsigned StrideH,
         unsigned StrideW, unsigned DilH, unsigned DilW, float *Output);

void convMEC(const float *Input, float *Kernel, float *Output, unsigned C,
             unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
             unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
             unsigned StrideH, unsigned StrideW, unsigned DilH, unsigned DilW);
//-------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ GEMM +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void packA(float *Block, float *&Pack, unsigned LDA, unsigned MC, unsigned KC);

void packB(float *Block, float *&Pack, unsigned LDB, unsigned KC, unsigned NC);

void gemm(float *A, float *B, float *C, unsigned M, unsigned K, unsigned N,
          unsigned LDA, unsigned LDB, unsigned LDC, float Alpha, float Beta);
//-------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Fused im2col & packing +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void im2colPackB(float *Input, float *&Pack, unsigned Ks, unsigned Ns, unsigned KC, unsigned NC, unsigned C, unsigned H, unsigned W, unsigned KH, unsigned KW, unsigned OW);
//-------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Yaconv +++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void yaconvPackB(float *Input, float *&Pack, unsigned Ks, unsigned Ns, unsigned KC, unsigned NC, unsigned C, unsigned H, unsigned W, unsigned KH, unsigned KW, unsigned OW);

void convGemm(float *Input, float *Kernel, float *Output, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW);
//-------------------------------------------------------------------------

#endif // _CONV_H_
