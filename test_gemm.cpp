#include "gemm.hpp"
#include "utils.h"
#include <blis.h>
#include <cblas.h>
#include <chrono>
#include <iostream>


using namespace std::chrono;


#define MAIN_DEBUG(expr) if (DEBUG == 1) {expr;}


int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: ./test_gemm M K N\n";
    return -1;
  }

  const unsigned M = atoi(argv[1]);
  const unsigned K = atoi(argv[2]);
  const unsigned N = atoi(argv[3]);

  // Gemm
  float *A = allocateFilledTensor(M * K);
  float *B = allocateFilledTensor(K * N);
  // float *A = allocateRandomTensor(M * K);
  // float *B = allocateRandomTensor(K * N);
  MAIN_DEBUG(
    printTensor(A, {M, K});
    printTensor(B, {K, N});
  )

  // Time variables
  high_resolution_clock::time_point t1, t2;
  double TempTime;
  std::vector<double> Times;

  // Output tensors
  std::vector<float *> Outputs;

  // GEMM variables
  float Alpha = 1.0, Beta = 0.0;
  const uint64_t Flops = 2 * (uint64_t)M * (uint64_t)K * (uint64_t)N;

#define RUN(f) \
  Outputs.push_back(allocateTensor(M * N)); \
  TempTime = 0.0; \
  for (unsigned i = 0; i < 1; ++i) { \
    t1 = high_resolution_clock::now(); \
    f; \
    t2 = high_resolution_clock::now(); \
    TempTime += duration_cast<duration<double>>(t2 - t1).count(); \
  } \
  Times.push_back(TempTime);

  // BLIS gemm
  RUN(bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &Alpha, A, K, 1, B, N, 1, &Beta, Outputs.back(), N, 1))

  // OpenBLAS gemm
  //RUN(cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, Alpha, A, K, B, N, Beta, Outputs.back(), N))

  // My gemm
  RUN(gemm(A, B, Outputs.back(), M, K, N, K, N, N, Alpha, Beta))

  // Print time for each run
  for (const auto &Time: Times)
    std::cout << Time << "\n";

  // Print tensors for each run
  MAIN_DEBUG(
    for (const auto &Output: Outputs)
      printTensor(Output, {M, N});
  )

  return tensorsEqual(Outputs, M * N) ? 0 : -1;
}
