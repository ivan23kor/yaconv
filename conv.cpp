#include "config.h"
#include "utils.h"
#include <algorithm>
#include <blis.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <string>


using namespace std;

#define CONV_DEBUG(expr) if (DEBUG == 1) {expr;}

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
  CONV_DEBUG(
    cout << "=== BufIm2col ===\n";
    printTensor(BufIm2col, {C * KH * KW, OH * OW});
    cout << string(80, '-') << "\n\n";
  )

  // Post-im2col GEMM
  {
    float alpha = 1.0, beta = 0.0;
    unsigned Mgemm = M;
    unsigned N = OH * OW;
    unsigned K = C * KH * KW;
    unsigned rsa = K;
    unsigned csa = 1;
    unsigned rsb = N;
    unsigned csb = 1;
    unsigned rsc = N;
    unsigned csc = 1;
    CONV_DEBUG(cout << Mgemm << " x " << K << " x " << N << " gemm\n";)
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, Mgemm, N, K, &alpha, Kernel,
              rsa, csa, BufIm2col, rsb, csb, &beta, Output, rsc, csc);
    CONV_DEBUG(
      cout << "=== OutputIm2col ===\n";
      printTensor(Output, {Mgemm, N});
      cout << string(80, '-') << "\n\n";
    )
  }
}

void conv_mec(const float *Input, float *Kernel, float *Output, unsigned C,
              unsigned H, unsigned W, unsigned M, unsigned KH, unsigned KW,
              unsigned OH, unsigned OW, unsigned PadH, unsigned PadW,
              unsigned StrideH, unsigned StrideW, unsigned DilH,
              unsigned DilW) {

  // MEC
  float *BufMec = mec(Input, C, H, W, KH, KW, 0, 0, 1, 1, 1, 1);
  CONV_DEBUG(
    cout << "=== BufMec ===\n";
    printTensor(BufMec, {OW, H * KW});
    cout << string(80, '-') << "\n\n";
  )

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
    CONV_DEBUG(cout << Mgemm << " x " << K << " x " << N << " gemm\n";)
    for (unsigned h = 0; h < OH; ++h) {
      bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, Mgemm, N, K, &alpha, Kernel,
                rsa, csa, BufMec + h * KW, rsb, csb, &beta, Output + h * OW,
                rsc, csc);
    }
    CONV_DEBUG(
      cout << "=== OutputMec ===\n";
      printTensor(Output, {M, OH * OW});
      cout << string(80, '-') << "\n\n";
    )
  }
}

#define MIN(a, b) a < b ? a : b
// Block is row-major
void packA(float *Block, float *&Pack, unsigned LDA, unsigned MC, unsigned KC) {
  unsigned To = 0;
  for (unsigned ic = 0; ic < MC; ic += BLOCK_MR) {
    unsigned MR = MIN(MC - ic, BLOCK_MR);
    for (unsigned k = 0; k < KC; ++k) {
      for (unsigned ir = 0; ir < MR; ++ir) {
        unsigned From = (ic + ir) * LDA + k;
        Pack[To++] = Block[From];
      }
      for (unsigned End = To + BLOCK_MR - MR; To != End; ++To)
        Pack[To] = 0.0;
    }
  }
}

// Block is row-major
void packB(float *Block, float *&Pack, unsigned LDB, unsigned KC, unsigned NC) {
  unsigned To = 0;
  for (unsigned jc = 0; jc < NC; jc += BLOCK_NR) {
    unsigned NR = MIN(NC - jc, BLOCK_NR);
    for (unsigned k = 0; k < KC; ++k) {
      for (unsigned jr = 0; jr < NR; ++jr) {
        unsigned From = jc + jr + LDB * k;
        Pack[To++] = Block[From];
      }
      for (unsigned End = To + BLOCK_NR - NR; To != End; ++To)
        Pack[To] = 0.0;
    }
  }
}

void sxpby(const float *restrict X, float *restrict Y, unsigned M, unsigned N,
            unsigned LDX, unsigned LDY, const float Beta) {
    if (Beta == 0.0)
      for(unsigned i = 0; i < M; ++i)
        for(unsigned j = 0; j < N; ++j)
          *(Y + i * LDY + j) = *(X + i * LDX + j);
    else
      for(unsigned i = 0; i < M; ++i)
        for(unsigned j = 0; j < N; ++j) {
          *(Y + i * LDY + j) *= Beta;
          *(Y + i * LDY + j) += *(X + i * LDX + j);
        }
}

void mm(float *A, float *B, float *C, unsigned M, unsigned K, unsigned N,
        unsigned LDA, unsigned LDB, unsigned LDC) {

  auto *APack = (float *)aligned_alloc(4096, BLOCK_MC * BLOCK_KC * sizeof(float));
  auto *BPack = (float *)aligned_alloc(4096, BLOCK_KC * BLOCK_NC * sizeof(float));
  auto *CBuff = (float *)aligned_alloc(4096, BLOCK_MR * BLOCK_NR * sizeof(float));

  auto *cntx = bli_gks_query_cntx();
  auto *data = new auxinfo_t;

  float Alpha = 1.0, Beta = 0.0, Zero = 0.0;
  for (unsigned jc = 0; jc < N; jc += BLOCK_NC) {

    unsigned NC = MIN(N - jc, BLOCK_NC);

    for (unsigned k = 0; k < K; k += BLOCK_KC) {

      unsigned KC = MIN(K - k, BLOCK_KC);

      Beta = k == 0 ? 0.0 : 1.0; // Accumulate or not

      packB(B + k * LDB + jc, BPack, LDB, KC, NC);

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
              bli_sgemm_haswell_asm_6x16(KC, &Alpha, Ar, Br, &Beta,
                  Cr, LDC, 1, data, cntx);
            else {
              bli_sgemm_haswell_asm_6x16(KC, &Alpha, Ar, Br, &Zero,
                  CBuff, BLOCK_NR, 1, data, cntx);
              sxpby(CBuff, Cr, MR, NR, BLOCK_NR, LDC, Beta);
            }
          }
        }
      }
    }
  }
}

// Input: C x H x W
void packInputAsB(float *Input, float *&Pack, unsigned Ks, unsigned Ns, unsigned KC, unsigned NC, unsigned C, unsigned H, unsigned W, unsigned KH, unsigned KW, unsigned OW) {
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

  auto *cntx = bli_gks_query_cntx();
  auto *data = new auxinfo_t;

  unsigned OH = H - KH + 1, OW = W - KW + 1;
  unsigned K = C * KH * KW;
  unsigned N = OH * OW;

  unsigned LDA = K, LDC = N;

  float Alpha = 1.0, Beta = 0.0, Zero = 0.0;

  for (unsigned jc = 0; jc < N; jc += BLOCK_NC) {

    unsigned NC = MIN(N - jc, BLOCK_NC);

    for (unsigned k = 0; k < K; k += BLOCK_KC) {

      unsigned KC = MIN(K - k, BLOCK_KC);

      Beta = k == 0 ? 0.0 : 1.0; // Accumulate

      packInputAsB(Input, BPack, k, jc, KC, NC, C, H, W, KH, KW, OW);

      for (unsigned ic = 0; ic < M; ic += BLOCK_MC) {

        unsigned MC = MIN(M - ic, BLOCK_MC);

        packA(Kernel + ic * LDA + k, APack, LDA, MC, KC);

        for (unsigned jr = 0; jr < NC; jr += BLOCK_NR) {

          unsigned NR = MIN(NC - jr, BLOCK_NR);

          for (unsigned ir = 0; ir < MC; ir += BLOCK_MR) {

            unsigned MR = MIN(MC - ir, BLOCK_MR);

            float *Ar = APack + ir * KC;
            float *Br = BPack + jr * KC;
            float *Cr = Output + (ic + ir) * LDC + jc + jr;

            if ((MR == BLOCK_MR) && (NR == BLOCK_NR))
              bli_sgemm_haswell_asm_6x16(KC, &Alpha, Ar, Br, &Beta,
                  Cr, LDC, 1, data, cntx);
            else {
              bli_sgemm_haswell_asm_6x16(KC, &Alpha, Ar, Br, &Zero,
                  CBuff, BLOCK_NR, 1, data, cntx);
              sxpby(CBuff, Cr, MR, NR, BLOCK_NR, LDC, Beta);
            }
          }
        }
      }
    }
  }
  CONV_DEBUG(
    cout << "=== OutputConvGemm ===\n";
    printTensor(Output, {M, N});
    cout << string(80, '-') << "\n\n";
  )
}
#undef MIN
