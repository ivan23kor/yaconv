#include "config.h"
#include "conv.h"
#include "utils.h"
#include <blis.h>
#include <chrono>
#include <iostream>


using namespace std;
using namespace std::chrono;

#define MAIN_DEBUG(expr) if (DEBUG == 1) {expr;}

int main(int argc, char **argv) {
  int ret = 0;

  // Calculate Output dimensions
  unsigned OH = H_ - KH_ + 1;
  unsigned OW = W_ - KW_ + 1;

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // +++ im2col and packing +++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Init Input
  float *Input = allocateFilledTensor(C_ * H_ * W_);
  MAIN_DEBUG(
    cout << "=== Input ===\n";
    printTensor(Input, {C_, H_, W_});
    cout << string(80, '-') << "\n\n";
  )

  // im2col
  float *im2colBuf = allocateFilledTensor(C_ * KH_ * KW_ * OH * OW);
  im2col_cpu(Input, C_, H_, W_, KH_, KW_, 0, 0, 1, 1, 1, 1, im2colBuf);
  MAIN_DEBUG(
    cout << "=== im2col ===\n";
    printTensor(im2colBuf, {C_ * KH_ * KW_, OH * OW});
    cout << string(80, '-') << "\n\n";
  )

  // Block sizes
  unsigned KC = C_ * KH_ * KW_, NC = OH * OW, NCBlock = NC;
  if (NC % BLOCK_NR)
    NCBlock += BLOCK_NR - NC % BLOCK_NR;
  float *BPackIm2col = allocateTensor(KC * NCBlock);
  MAIN_DEBUG(
    cout << KC << " x " << NCBlock << "\n";
  )

  // Pack im2col
  packB(im2colBuf, BPackIm2col, NC, KC, NC);
  MAIN_DEBUG(
    cout << "=== im2col, then pack ===\n";
    printTensor(BPackIm2col, {KC * NCBlock / BLOCK_NR, BLOCK_NR});
    cout << string(80, '-') << "\n\n";
  )

  // Fused im2col+packB
  float *BPack = allocateTensor(KC * NC);
  packInputAsB(Input, BPack, 0, 0, KC, NC, C_, H_, W_, KH_, KW_, OW);
  MAIN_DEBUG(
    cout << "=== im2col&pack ===\n";
    printTensor(BPack, {KC * NCBlock / BLOCK_NR, BLOCK_NR});
    cout << string(80, '-') << "\n\n";
  )
  // --------------------------------------------------------------------------
  // --------------------------------------------------------------------------
  // --------------------------------------------------------------------------

  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // +++ Convolution +++
  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // Init Kernel
  // float *Kernel = allocateFilledTensor(M_ * C_ * KH_ * KW_);
  // MAIN_DEBUG(
  //   cout << "=== Kernel ===\n";
  //   printTensor(Kernel, {M_, C_ * KH_ * KW_});
  //   cout << string(80, '-') << "\n\n";
  // )

  // // Init Input
  // float *Input = allocateFilledTensor(C_ * H_ * W_);
  // MAIN_DEBUG(
  //   cout << "=== Input ===\n";
  //   printTensor(Input, {C_, H_, W_});
  //   cout << string(80, '-') << "\n\n";
  // )

  // // Convolution with im2col
  // float *OutputIm2col = allocateTensor(M_ * OH * OW);
  // conv_im2col(Input, Kernel, OutputIm2col, C_, H_, W_, M_, KH_, KW_, OH, OW, 0,
  //             0, 1, 1, 1, 1);

  // // Convolution with fused im2col+packing
  // float *OutputConvGemm = allocateTensor(M_ * OH * OW);
  // convGemm(Input, Kernel, OutputConvGemm, C_, H_, W_, M_, KH_, KW_);

  // if (!tensorsEqual(OutputIm2col, OutputConvGemm, M_ * OH * OW))
  //   ret = -1;

  // // // Convolution with MEC
  // // float *OutputMec = allocateTensor(M_ * OH * OW);
  // // conv_mec(Input, Kernel, OutputMec, C_, H_, W_, M_, KH_, KW_, OH, OW, 0, 0, 1,
  // //          1, 1, 1);
  // // --------------------------------------------------------------------------
  // // --------------------------------------------------------------------------
  // // --------------------------------------------------------------------------



  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // +++ GEMM +++
  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // if (argc != 4) {
  //   cerr << "Usage: ./yaconv M K N\n";
  //   return -1;
  // }

  // unsigned M = atoi(argv[1]);
  // unsigned K = atoi(argv[2]);
  // unsigned N = atoi(argv[3]);

  // // Gemm
  // float *A = allocateFilledTensor(M * K);
  // float *B = allocateFilledTensor(K * N);
  // // float *A = allocateRandomTensor(M * K);
  // // float *B = allocateRandomTensor(K * N);
  // float *CBLIS = allocateTensor(M * N);
  // float *CMine = allocateTensor(M * N);
  // MAIN_DEBUG(
  //   printTensor(A, {M, K});
  //   printTensor(B, {K, N});
  // )

  // float Alpha = 1.0, Beta = 0.0;
  // high_resolution_clock::time_point t1, t2;

  // // BLIS gemm
  // t1 = high_resolution_clock::now();
  // bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &Alpha, A, K, 1, B, N, 1, &Beta, CBLIS, N, 1);
  // t2 = high_resolution_clock::now();
  // double BLIS = duration_cast<duration<double>>(t2 - t1).count();

  // // My gemm
  // t1 = high_resolution_clock::now();
  // mm(A, B, CMine, M, K, N, K, N, N);
  // t2 = high_resolution_clock::now();
  // double Mine = duration_cast<duration<double>>(t2 - t1).count();

  // cout << BLIS / Mine << "\n";

  // if (!tensorsEqual(CMine, CBLIS, M * N))
  //   ret = -1;

  // MAIN_DEBUG(
  //   printTensor(CMine, {M, N});
  //   printTensor(CBLIS, {M, N});
  // )
  // // --------------------------------------------------------------------------
  // // --- GEMM ---
  // // --------------------------------------------------------------------------

  return ret;
}
