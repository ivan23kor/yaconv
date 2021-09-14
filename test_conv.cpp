#include "conv.hpp"
#include "yaconv.hpp"
#include "utils.hpp"
#include <blis.h>
#include <cblas.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;

// This is to allow convolution algorithms to add their timed portions to
// this vector
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

  float *Kernel = allocateFilledTensor(M * C * KH * KW);
  float *Input = allocateFilledTensor(C * H * W);
  IF_DEBUG(printTensor(Kernel, {M, C, KH * KW}))
  IF_DEBUG(printTensor(Input, {C, H, W}))

  // Output tensors
  std::vector<float *> Outputs;

  // Time variables
  high_resolution_clock::time_point t1, t2;
  double TempTime;

#define RUN_CONV(f) \
  RUN(M * OH * OW, f, Repeat)

  // clang-format off
  // // Convolution with im2col
  RUN_CONV(convIm2col(Input, Kernel, Outputs.back(), C, H, W, M, KH, KW, OH, OW, 0, 0, 1, 1, 1, 1))

  // // Convolution with fused im2col+packing
  // RUN_CONV(Outputs.back() = convGemm(Input, Kernel, C, H, W, M, KH, KW))

  // // Convolution with MEC for NCHW format
  // RUN_CONV(Outputs.back() = convMecNCHW(Input, Kernel, C, H, W, M, KH, KW))

  // Yaconv
  RUN_CONV(Outputs.back() = yaconv(Input, Kernel, C, H, W, M, KH, KW, OH, OW, 0, 0, 1, 1, 1, 1))
  // clang-format on

  // Print tensors for each run
  IF_DEBUG(for (const auto &Output
                  : Outputs) printTensor(Output, {M, OH * OW});)

  // Print times for each run
  for (const auto &Time : Times)
    std::cout << Time << "\n";

  return tensorsEqual(Outputs, M * OH * OW) ? 0 : -1;
}
