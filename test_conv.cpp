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

void convertCHWToHWC(float *CHW, float *HWC, unsigned C, unsigned H, unsigned W) {
  for (unsigned c = 0; c < C; ++c)
    for (unsigned h = 0; h < H; ++h)
      for (unsigned w = 0; w < W; ++w)
        HWC[h * W * C + w * C + c] = CHW[c * H * W + h * W + w];
}

void convertMCHWToMHWC(float *MCHW, float *MHWC, unsigned M, unsigned C, unsigned H, unsigned W) {
  for (unsigned m = 0; m < M; ++m)
    convertCHWToHWC(MCHW + m * C * H * W, MHWC + m * H * W * C, C, H, W);
}

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
  const unsigned PH = atoi(argv[7]);
  const unsigned PW = atoi(argv[8]);
  const unsigned SH = atoi(argv[9]);
  const unsigned SW = atoi(argv[10]);
  const unsigned Repeat = argc > 11 ? atoi(argv[11]) : 1;

  const unsigned OH = (H - FH + 2 * PH) / SH + 1;
  const unsigned OW = (W - FW + 2 * PW) / SW + 1;

  // Create two input tensors - NCHW and NHWC
  auto *InputCHW = allocateFilledTensor(C * H * W);
  auto *InputHWC = allocateFilledTensor(H * W * C);
  convertCHWToHWC(InputCHW, InputHWC, C, H, W);

  // Create two filter tensors - MCHW and MHWC
  auto *FilterMCHW = allocateFilledTensor(M * C * FH * FW);
  auto *FilterMHWC = allocateFilledTensor(M * FH * FW * C);
  convertMCHWToMHWC(FilterMCHW, FilterMHWC, M, C, FH, FW);

  // Print input and filter tensors in debug mode
  IF_DEBUG(printTensor(InputCHW, {C, H, W}))
  IF_DEBUG(printTensor(FilterMCHW, {M, C, FH * FW}))

  // Output tensors
  std::vector<float *> Outputs;

  // Time variables
  high_resolution_clock::time_point t1, t2;
  double TempTime;

#define RUN_CONV(f) \
  RUN(M * OH * OW, f, Repeat)

  // clang-format off
  // // Convolution with im2col
  // RUN_CONV(convIm2col(InputCHW, FilterMCHW, Outputs.back(), C, H, W, M, FH, FW, OH, OW, PH, PW, SH, SW))

  // Yaconv
  RUN_CONV(yaconv(InputHWC, FilterMHWC, Outputs.back(), C, H, W, M, FH, FW, SH, SW, PH, PW))
  // clang-format on

  // Print tensors for each run
  IF_DEBUG(for (const auto &Output
                  : Outputs) printTensor(Output, {M, OH * OW});)

  // Print times for each run
  // for (const auto &Time : Times)
  //   std::cout << Time << "\n";
  // std::cout << Im2colTime / Repeat << "," << GEMMTime / Repeat << "\n";

  free(InputCHW);
  free(InputHWC);
  free(FilterMCHW);
  free(FilterMHWC);
  for (const auto &Output: Outputs)
    free(Output);
  return 0;
  return tensorsEqual(Outputs, M * OH * OW) ? 0 : -1;
}
