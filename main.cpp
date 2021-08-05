#include "utils.h"
#include "conv.h"
#include <blis.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

#ifndef C_
#define C_ 1
#endif

#ifndef H_
#define H_ 6
#endif

#ifndef W_
#define W_ 6
#endif

#ifndef M_
#define M_ 1
#endif

#ifndef KH_
#define KH_ 3
#endif

#ifndef KW_
#define KW_ 3
#endif

int main() {
  // // Init Kernel
  // float *Kernel = allocateSerialTensor(M_ * C_ * KH_ * KW_);
  // cout << "=== Kernel ===\n";
  // printTensor(Kernel, {M_, C_ * KH_ * KW_});
  // cout << string(80, '-') << "\n\n";

  // // Init Input
  // float *Input = allocateSerialTensor(C_ * H_ * W_);
  // cout << "=== Input ===\n";
  // printTensor(Input, {C_, H_, W_});
  // cout << string(80, '-') << "\n\n";

  // // Calculate Output dimensions
  // unsigned OH = H_ - KH_ + 1;
  // unsigned OW = W_ - KW_ + 1;

  // // Convolution with im2col
  // float *OutputIm2col = allocateTensor(M_ * OH * OW);
  // conv_im2col(Input, Kernel, OutputIm2col, C_, H_, W_, M_, KH_, KW_, OH, OW, 0,
  //             0, 1, 1, 1, 1);

  // // Convolution with MEC
  // float *OutputMec = allocateTensor(M_ * OH * OW);
  // conv_mec(Input, Kernel, OutputMec, C_, H_, W_, M_, KH_, KW_, OH, OW, 0, 0, 1,
  //          1, 1, 1);

  // Convolution with convGemm
  unsigned M = 16800, K = 2560, N = 4080;
  float *A = allocateAndFillTensor(M * K);
  float *B = allocateAndFillTensor(K * N);
  float *C = allocateTensor(M * N);
  float *CAns = allocateTensor(M * N);
  // printTensor(A, {M, K});
  // printTensor(B, {K, N});
  // printTensor(C, {M, N});

  high_resolution_clock::time_point t1, t2;

  // Lib gemm
  float alpha = 1.0, beta = 0.0;
  t1 = high_resolution_clock::now();
  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &alpha, A, K, 1, B, N, 1, &beta, CAns, N, 1);
  t2 = high_resolution_clock::now();
  double BLIS = duration_cast<duration<double>>(t2 - t1).count();

  // My gemm
  t1 = high_resolution_clock::now();
  mm(A, B, C, M, K, N, K, N, N);
  t2 = high_resolution_clock::now();
  double Mine = duration_cast<duration<double>>(t2 - t1).count();

  cout << Mine / BLIS << "\n";

  if (tensorsEqual(C, CAns, M * N))
    cout << "Good\n";
  else
    cout << "Bad\n";
  // mm(KH_, KW_, C_, M_, 1.0, Kernel, H_, W_, 1, 1, 1, Input, 0.0, OutputConvGemm);

  // Check conv outputs match
  int ret = 0;
  // if (!tensorsEqual(OutputIm2col, OutputMec, M_ * OH * OW)) ret = -1;
  // if (!tensorsEqual(OutputIm2col, OutputConvGemm, M_ * OH * OW)) ret = -1;

  return ret;
}
