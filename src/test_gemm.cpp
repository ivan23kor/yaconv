#include "gemm.hpp"
#include "utils.hpp"
#include <blis.h>
#ifdef OPENBLAS_GEMM
#include <cblas.h>
#endif
#include <chrono>
#include <iostream>

using namespace std::chrono;

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: ./test_gemm M K N [Repeat]\n";
    return -1;
  }

  const int M = atoi(argv[1]);
  const int K = atoi(argv[2]);
  const int N = atoi(argv[3]);
  const int Repeat = argc > 4 ? atoi(argv[4]) : 1;

  // Gemm
  float *A = allocateRandomTensor(M * K);
  float *B = allocateRandomTensor(K * N);

  // Output tensors
  std::vector<float *> Outputs;

  // GEMM variables
  float Alpha = 1.0, Beta = 0.0;

  // Time variables
  high_resolution_clock::time_point t1, t2;
  double TempTime;
  std::vector<double> Times;

#define RUN(f)                                                                 \
  Outputs.push_back(alignedAlloc(M * N));                                      \
  f;                                                                           \
  TempTime = 0.0;                                                              \
  for (int i = 0; i < Repeat; ++i) {                                           \
    flushCache();                                                              \
    t1 = high_resolution_clock::now();                                         \
    f;                                                                         \
    t2 = high_resolution_clock::now();                                         \
    TempTime += duration_cast<duration<double>>(t2 - t1).count();              \
  }                                                                            \
  Times.push_back(TempTime / Repeat);

  // BLIS gemm
#ifdef OPENBLAS_GEMM
  RUN(cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, Alpha, A, K, B, N, Beta, Outputs.back(), M))
#endif

  // BLIS gemm
  RUN(bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &Alpha, A, K, 1, B, N, 1, &Beta, Outputs.back(), N, 1))

  // Custom gemm
  // RUN(gemm(A, B, Outputs.back(), M, K, N, K, N, N, Alpha, Beta))

  // Print tensors after each run
  // for (const auto &Output : Outputs)
  //   printTensor(Output, {M, N});

  // Print time for each run
  for (const auto &Time : Times)
    std::cout << Time << "\n";

  std::cout << "Max relative diff: " << maxRelativeDiff(Outputs, M * N) << "\n";

  return 0;
}
