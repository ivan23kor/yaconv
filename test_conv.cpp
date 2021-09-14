#include "conv.hpp"
#include "utils.h"
#include <blis.h>
#include <cblas.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;

#define MAIN_DEBUG(expr)                                                       \
  if (DEBUG == 1) {                                                            \
    expr;                                                                      \
  }

// Algorithms add their timed portions to this vector
// main() adds total execution runtime
std::vector<double> Times;

int main(int argc, char **argv) {
  // For now, N == 1
  if (argc < 7) {
    std::cerr << "Usage: ./test_conv M KH KW C H W [Repeat]\n";
    return -1;
  }

  const unsigned M = atoi(argv[1]);
  const unsigned KH = atoi(argv[2]);
  const unsigned KW = atoi(argv[3]);
  const unsigned C = atoi(argv[4]);
  const unsigned H = atoi(argv[5]);
  const unsigned W = atoi(argv[6]);
  const unsigned Repeat = argc > 7 ? atoi(argv[7]) : 1;

  const unsigned OH = H - KH + 1;
  const unsigned OW = W - KW + 1;

  // Init Kernel
  float *Kernel = allocateFilledTensor(M * C * KH * KW);
  MAIN_DEBUG(printTensor(Kernel, {M, C, KH * KW}))

  // Init Input
  float *Input = allocateFilledTensor(C * H * W);
  MAIN_DEBUG(printTensor(Input, {C, H, W}))

  // Time variables
  high_resolution_clock::time_point t1, t2;
  double TempTime;

  // Output tensors
  std::vector<float *> Outputs;

#define RUN(f)                                                                 \
  Outputs.push_back(alignedAlloc(M * OH * OW));                                \
  TempTime = 0.0;                                                              \
  for (unsigned i = 0; i < Repeat; ++i) {                                      \
    t1 = high_resolution_clock::now();                                         \
    f;                                                                         \
    t2 = high_resolution_clock::now();                                         \
    TempTime += duration_cast<duration<double>>(t2 - t1).count();              \
  }                                                                            \
  Times.push_back(TempTime);

  // // Convolution with im2col
  // RUN(convIm2col(Input, Kernel, Outputs.back(), C, H, W, M, KH, KW, OH, OW, 0, 0, 1, 1, 1, 1))

  // // Convolution with fused im2col+packing
  // RUN(Outputs.back() = convGemm(Input, Kernel, C, H, W, M, KH, KW))

  // // Convolution with MEC for NCHW format
  // RUN(Outputs.back() = convMecNCHW(Input, Kernel, C, H, W, M, KH, KW))

  // Yaconv
  RUN(Outputs.back() = yaconv(Input, Kernel, C, H, W, M, KH, KW, OH, OW, 0, 0, 1, 1, 1, 1))

  // Print times for each run
  for (const auto &Time : Times)
    std::cout << Time << "\n";

  // Print tensors for each run
  MAIN_DEBUG(for (const auto &Output
                  : Outputs) printTensor(Output, {M, OH * OW});)

  return tensorsEqual(Outputs, M * OH * OW) ? 0 : -1;
}
