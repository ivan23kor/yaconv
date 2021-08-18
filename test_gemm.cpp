#include "gemm.h"
#include "utils.h"
#include <blis.h>
#include <cblas.h>
#include <chrono>
#include <iostream>


using namespace std;
using namespace std::chrono;


#define MAIN_DEBUG(expr) if (DEBUG == 1) {expr;}


int main(int argc, char **argv) {
  if (argc != 4) {
    cerr << "Usage: ./test_gemm M K N\n";
    return -1;
  }

  const unsigned M = atoi(argv[1]);
  const unsigned K = atoi(argv[2]);
  const unsigned N = atoi(argv[3]);
  const uint64_t Flops = 2 * (uint64_t)M * (uint64_t)K * (uint64_t)N;

  // Gemm
  // float *A = allocateFilledTensor(M * K);
  // float *B = allocateFilledTensor(K * N);
  float *A = allocateRandomTensor(M * K);
  float *B = allocateRandomTensor(K * N);
  float *CBLIS = allocateTensor(M * N);
  float *CBLAS = allocateTensor(M * N);
  float *CMine = allocateTensor(M * N);
  MAIN_DEBUG(
    printTensor(A, {M, K});
    printTensor(B, {K, N});
  )

  float Alpha = 1.0, Beta = 0.0;
  high_resolution_clock::time_point t1, t2;

  // BLIS gemm
  t1 = high_resolution_clock::now();
  bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &Alpha, A, K, 1, B, N, 1, &Beta, CBLIS, N, 1);
  t2 = high_resolution_clock::now();
  double BLISTime = duration_cast<duration<double>>(t2 - t1).count();

  // OpenBLAS gemm
  t1 = high_resolution_clock::now();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, Alpha, A, K, B, N, Beta, CBLAS, N);
  t2 = high_resolution_clock::now();
  double BLASTime = duration_cast<duration<double>>(t2 - t1).count();

  // My gemm
  t1 = high_resolution_clock::now();
  gemm(A, B, CMine, M, K, N, K, N, N, Alpha, Beta);
  t2 = high_resolution_clock::now();
  double MineTime = duration_cast<duration<double>>(t2 - t1).count();

  cout << "BLIS: " << Flops / BLISTime * 1e-9 << " GFLOPS\n";
  cout << "BLAS: " << Flops / BLASTime * 1e-9 << " GFLOPS\n";
  cout << "Mine: " << Flops / MineTime * 1e-9 << " GFLOPS\n";

  MAIN_DEBUG(
    cout << "BLIS:\n";
    printTensor(CBLIS, {M, N});
    cout << "OpenBLAS:\n";
    printTensor(CBLAS, {M, N});
    cout << "Mine:\n";
    printTensor(CMine, {M, N});
  )

  return tensorsEqual(CBLIS, CMine, M * N) ? 0 : -1;
}
