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
  unsigned OH_ = H_ - KH_ + 1;
  unsigned OW_ = W_ - KW_ + 1;

  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // +++ im2col and packing +++
  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // Init Input
  // float *Input = allocateFilledTensor(C_ * H_ * W_);
  // MAIN_DEBUG(
  //   cout << "=== Input ===\n";
  //   printTensor(Input, {C_, H_, W_});
  //   cout << string(80, '-') << "\n\n";
  // )

  // // im2col
  // float *im2colBuf = allocateFilledTensor(C_ * KH_ * KW_ * OH_ * OW_);
  // im2col_cpu(Input, C_, H_, W_, KH_, KW_, 0, 0, 1, 1, 1, 1, im2colBuf);
  // MAIN_DEBUG(
  //   cout << "=== im2col ===\n";
  //   printTensor(im2colBuf, {C_ * KH_ * KW_, OH_ * OW_});
  //   cout << string(80, '-') << "\n\n";
  // )

  // // Block sizes
  // unsigned KC = C_ * KH_ * KW_, NC = OH_ * OW_, NCBlock = NC;
  // if (NC % BLOCK_NR)
  //   NCBlock += BLOCK_NR - NC % BLOCK_NR;
  // float *BPackIm2col = allocateTensor(KC * NCBlock);
  // MAIN_DEBUG(
  //   cout << KC << " x " << NCBlock << "\n";
  // )

  // // Pack im2col
  // packB(im2colBuf, BPackIm2col, NC, KC, NC);
  // MAIN_DEBUG(
  //   cout << "=== im2col, then pack ===\n";
  //   printTensor(BPackIm2col, {KC * NCBlock / BLOCK_NR, BLOCK_NR});
  //   cout << string(80, '-') << "\n\n";
  // )

  // // Fused im2col+packB
  // float *BPack = allocateTensor(KC * NCBlock);
  // im2colPackB(Input, BPack, 0, 0, KC, NC, C_, H_, W_, KH_, KW_, OW_);
  // MAIN_DEBUG(
  //   cout << "=== im2col&pack ===\n";
  //   printTensor(BPack, {KC * NCBlock / BLOCK_NR, BLOCK_NR});
  //   cout << string(80, '-') << "\n\n";
  // )
  // // ------------------------------------------------------------------------

  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // +++ MEC and packing +++
  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // Init Input
  // float *Input = allocateFilledTensor(C_ * H_ * W_);
  // MAIN_DEBUG(
  //   cout << "=== Input ===\n";
  //   printTensor(Input, {C_, H_, W_});
  //   cout << string(80, '-') << "\n\n";
  // )

  // // MEC
  // float *MECBuf = allocateFilledTensor(C_ * W_ * KW_ * OW_);
  // mec(Input, C_, H_, W_, KH_, KW_, 0, 0, 1, 1, 1, 1, MECBuf);
  // MAIN_DEBUG(
  //   cout << "=== MEC ===\n";
  //   printTensor(MECBuf, {C_ * H_ * KW_, OW_});
  //   cout << string(80, '-') << "\n\n";
  // )

  // // Block sizes
  // unsigned KC = C_ * H_ * KW_, NC = OW_, NCBlock = NC;
  // if (NC % BLOCK_NR)
  //   NCBlock += BLOCK_NR - NC % BLOCK_NR;
  // float *BPackMEC = allocateTensor(KC * NCBlock);
  // MAIN_DEBUG(
  //   cout << KC << " x " << NCBlock << "\n";
  // )

  // // Pack MECBuf
  // packB(MECBuf, BPackMEC, NC, KC, NC);
  // MAIN_DEBUG(
  //   cout << "=== MEC, then pack ===\n";
  //   printTensor(BPackMEC, {KC * NCBlock / BLOCK_NR, BLOCK_NR});
  //   cout << string(80, '-') << "\n\n";
  // )

  // // Fused MEC & pack (yaconv)
  // float *BPack = allocateTensor(KC * NCBlock);
  // yaconvPackB(Input, BPack, 0, 0, KC, NC, C_, H_, W_, KH_, KW_, OW_);
  // MAIN_DEBUG(
  //   cout << "=== Fused MEC & pack (yaconv) ===\n";
  //   printTensor(BPack, {KC * NCBlock / BLOCK_NR, BLOCK_NR});
  //   cout << string(80, '-') << "\n\n";
  // )
  // // ------------------------------------------------------------------------

  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // +++ Convolution +++
  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
  // float *OutputIm2col = allocateTensor(M_ * OH_ * OW_);
  // convIm2col(Input, Kernel, OutputIm2col, C_, H_, W_, M_, KH_, KW_, OH_, OW_, 0,
  //             0, 1, 1, 1, 1);

  // // // Convolution with fused im2col+packing
  // // float *OutputConvGemm = allocateTensor(M_ * OH_ * OW_);
  // // convGemm(Input, Kernel, OutputConvGemm, C_, H_, W_, M_, KH_, KW_);

  // // Convolution with MEC
  // float *OutputMEC = allocateTensor(M_ * OH_ * OW_);
  // convMEC(Input, Kernel, OutputMEC, C_, H_, W_, M_, KH_, KW_, OH_, OW_, 0, 0, 1,
  //         1, 1, 1);

  // if (!tensorsEqual(OutputIm2col, OutputMEC, M_ * OH_ * OW_))
  //   ret = -1;
  // // ------------------------------------------------------------------------



  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // // +++ GEMM +++
  // // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // if (argc != 4) {
  //   cerr << "Usage: ./yaconv M K N\n";
  //   return -1;
  // }

  // const unsigned M = atoi(argv[1]);
  // const unsigned K = atoi(argv[2]);
  // const unsigned N = atoi(argv[3]);
  // const unsigned Flops = 2 * M * K * N;

  // // Gemm
  // // float *A = allocateFilledTensor(M * K);
  // // float *B = allocateFilledTensor(K * N);
  // float *A = allocateRandomTensor(M * K);
  // float *B = allocateRandomTensor(K * N);
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
  // gemm(A, B, CMine, M, K, N, K, N, N, Alpha, Beta);
  // t2 = high_resolution_clock::now();
  // double Mine = duration_cast<duration<double>>(t2 - t1).count();

  // cout << "BLIS: " << Flops / BLIS * 1e-9 << " GFLOPS\n";
  // cout << "Mine: " << Flops / Mine * 1e-9 << " GFLOPS\n";

  // if (!tensorsEqual(CMine, CBLIS, M * N))
  //   ret = -1;

  // MAIN_DEBUG(
  //   printTensor(CMine, {M, N});
  //   printTensor(CBLIS, {M, N});
  // )
  // // ------------------------------------------------------------------------

  return ret;
}
