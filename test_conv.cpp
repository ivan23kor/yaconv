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

extern double PackImageTime, PackFilterTime;

int main(int argc, char **argv) {
  // For now, N == 1
  if (argc < 11) {
    std::cerr << "Usage: ./test_conv C H W M FH FW SH SW PH PW [Repeat]\n";
    return -1;
  }

  const unsigned C = atoi(argv[1]);
  const unsigned H = atoi(argv[2]);
  const unsigned W = atoi(argv[3]);
  const unsigned M = atoi(argv[4]);
  const unsigned FH = atoi(argv[5]);
  const unsigned FW = atoi(argv[6]);
  const unsigned SH = atoi(argv[7]);
  const unsigned SW = atoi(argv[8]);
  const unsigned PH = atoi(argv[9]);
  const unsigned PW = atoi(argv[10]);
  const unsigned Repeat = argc > 11 ? atoi(argv[11]) : 1;

  const unsigned OH = (H - FH + 2 * PH) / SH + 1;
  const unsigned OW = (W - FW + 2 * PW) / SW + 1;
  std::cout << "Output: [" << M << "] x [" << OH << " * " << OW << "]\n";

  float *Input = allocateFilledTensor(C * H * W);
  float *Filter = allocateFilledTensor(M * C * FH * FW);
  IF_DEBUG(printTensor(Input, {C, H, W}))
  IF_DEBUG(printTensor(Filter, {M, C, FH * FW}))

  // Output tensors
  std::vector<float *> Outputs;

  // Time variables
  high_resolution_clock::time_point t1, t2;
  double TempTime;

#define RUN_CONV(f) \
  RUN(M * OH * OW, f, Repeat)

  // clang-format off
  // // Convolution with im2col
  // RUN_CONV(convIm2col(Input, Filter, Outputs.back(), C, H, W, M, FH, FW, OH, OW, PH, PW, SH, SW))

  // Yaconv
  RUN_CONV(yaconv(Input, Filter, Outputs.back(), C, H, W, M, FH, FW, SH, SW, PH, PW))

  // // Convolution with fused im2col+packing
  // RUN_CONV(Outputs.back() = convGemm(Input, Filter, C, H, W, M, FH, FW))
  // clang-format on

  // // Print tensors for each run
  // IF_DEBUG(for (const auto &Output
  //                 : Outputs) printTensor(Output, {M, OH * OW});)

  // Print times for each run
  for (const auto &Time : Times)
    std::cout << Time << "\n";
  std::cout << "PackImageTime: " << PackImageTime / Repeat << "\n";
  std::cout << "PackFilterTime: " << PackFilterTime / Repeat << "\n";

  return 0; //tensorsEqual(Outputs, M * OH * OW) ? 0 : -1;
}
