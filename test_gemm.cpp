#include "gemm.hpp"
#include "utils.hpp"
#include <blis.h>
#include <cblas.h>
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
  // printTensor(A, {M, K}); printTensor(B, {K, N});

  // Output tensors
  std::vector<float *> Outputs;

  // GEMM variables
  float Alpha = 1.0, Beta = 0.0;
  const uint64_t Flops = 2 * (uint64_t)M * (uint64_t)K * (uint64_t)N;

  // Time variables
  high_resolution_clock::time_point t1, t2;
  double TempTime;
  std::vector<double> Times;

#define RUN(f)                                                                 \
  Outputs.push_back(alignedAlloc(M * N));                                      \
  TempTime = 0.0;                                                              \
  for (int i = 0; i < Repeat; ++i) {                                           \
    flushCache();                                                              \
    t1 = high_resolution_clock::now();                                         \
    f;                                                                         \
    t2 = high_resolution_clock::now();                                         \
    TempTime += duration_cast<duration<double>>(t2 - t1).count();              \
  }                                                                            \
  Times.push_back(TempTime / Repeat);

  // clang-format off
  // BLIS gemm
  RUN(bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &Alpha, A, K, 1, B, N, 1, &Beta, Outputs.back(), N, 1))

  // My gemm
  RUN(gemm(A, B, Outputs.back(), M, K, N, K, N, N, Alpha, Beta))
  // clang-format on

  // Print tensors for each run
  // for (const auto &Output : Outputs) printTensor(Output, {M, N});

  // Print time for each run
  for (const auto &Time : Times)
    std::cout << Time << "\n";

  return tensorsEqual(Outputs, M * N) ? 0 : -1;
}
