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

int main(int argc, char **argv) {
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

  if (argc != 4) {
    cerr << "Usage: ./yaconv M K N\n";
    return -1;
  }

  unsigned M = atoi(argv[1]);
  unsigned K = atoi(argv[2]);
  unsigned N = atoi(argv[3]);

  // Convolution with convGemm
  // float *A = allocateFilledTensor(M * K);
  // float *B = allocateFilledTensor(K * N);
  float *A = allocateRandomTensor(M * K);
  float *B = allocateRandomTensor(K * N);
  float *CBLIS = allocateTensor(M * N);
  float *CMine = allocateTensor(M * N);
  // printTensor(A, {M, K});
  // printTensor(B, {K, N});

  float Alpha = 1.0, Beta = 0.0;
  high_resolution_clock::time_point t1, t2;

  // BLIS gemm
  t1 = high_resolution_clock::now();
  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &Alpha, A, K, 1, B, N, 1, &Beta, CBLIS, N, 1);
  t2 = high_resolution_clock::now();
  double BLIS = duration_cast<duration<double>>(t2 - t1).count();

  // My gemm
  t1 = high_resolution_clock::now();
  mm(A, B, CMine, M, K, N, K, N, N);
  t2 = high_resolution_clock::now();
  double Mine = duration_cast<duration<double>>(t2 - t1).count();

  cout << BLIS / Mine << "\n";

  if (tensorsEqual(CMine, CBLIS, M * N))
    cout << "Good\n";
  else
    cout << "Bad\n";
  if (M * N < 1000) {
    printTensor(CMine, {M, N});
    printTensor(CBLIS, {M, N});
  }

  // Check conv outputs match
  int ret = 0;
  // if (!tensorsEqual(OutputIm2col, OutputMec, M_ * OH * OW)) ret = -1;
  // if (!tensorsEqual(OutputIm2col, OutputConvGemm, M_ * OH * OW)) ret = -1;

  return ret;
}
