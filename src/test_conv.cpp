#include "conv.hpp"
#include "yaconv.hpp"
#include "utils.hpp"
#include <blis.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;
extern double YaConvPack1, YaConvPack2, YaConvComp;
extern double Im2colCopy, Im2colComp;

void convertCHWToHWC(float *CHW, float *HWC, int C, int H, int W) {
  for (int c = 0; c < C; ++c)
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        HWC[h * W * C + w * C + c] = CHW[c * H * W + h * W + w];
}

void convertMCHWToMHWC(float *MCHW, float *MHWC, int M, int C, int H, int W) {
  for (int m = 0; m < M; ++m)
    convertCHWToHWC(MCHW + m * C * H * W, MHWC + m * H * W * C, C, H, W);
}

float *yaconvExtraOutput(int H, int W, int FH, int FW, int PW, int OW, int M) {
  int WholeH = H % 16 == 0 ? H : H + 16 - H % 16;
  int ExtraBefore = ((FH - 1) * OW + PW) * M;
  int OutputAndAfter = WholeH * OW * M + (W + PW - FW + 1) * M;
  // std::cout << ExtraBefore << " before and " << OutputAndAfter << " for output and after\n";

  auto *Ret = alignedAlloc(ExtraBefore + OutputAndAfter * 10);
  return Ret + ExtraBefore;
}

int main(int argc, char **argv) {
  // For now, N == 1
  if (argc < 11) {
    std::cerr << "Usage: ./test_conv C H W M FH FW SH SW PH PW [Repeat]\n";
    return -1;
  }

  const int C = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int W = atoi(argv[3]);
  const int M = atoi(argv[4]);
  const int FH = atoi(argv[5]);
  const int FW = atoi(argv[6]);
  const int PH = atoi(argv[7]);
  const int PW = atoi(argv[8]);
  const int SH = atoi(argv[9]);
  const int SW = atoi(argv[10]);
  const int Repeat = argc > 11 ? atoi(argv[11]) : 1;

  const int OH = (H - FH + 2 * PH) / SH + 1;
  const int OW = (W - FW + 2 * PW) / SW + 1;

  // Create two input tensors - NCHW and NHWC
  auto *InputCHW = allocateRandomTensor(C * H * W);
  auto *InputHWC = allocateRandomTensor(H * W * C);
  convertCHWToHWC(InputCHW, InputHWC, C, H, W);

  // Create two filter tensors - MCHW and MHWC
  auto *FilterMCHW = allocateRandomTensor(M * C * FH * FW);
  auto *FilterMHWC = allocateRandomTensor(M * FH * FW * C);
  convertMCHWToMHWC(FilterMCHW, FilterMHWC, M, C, FH, FW);

  // Print Input and Filter tensors
  // printTensor(InputCHW, {C, H, W});
  // printTensor(InputHWC, {H, W, C});
  // printTensor(FilterMCHW, {M, C, FH, FW});
  // printTensor(FilterMHWC, {M, FH, FW, C});

  // Output tensors
  std::vector<float *> Outputs;

  // Time variables
  high_resolution_clock::time_point t1, t2;
  std::vector<double> Times;
  double TempTime;

#define RUN(f)                                                                 \
  f;                                                                           \
  YaConvPack1 = YaConvPack2 = YaConvComp = 0.0;                                \
  Im2colCopy = Im2colComp = 0.0;                                \
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
  // Convolution with im2col
  auto *im2colOutputMHW = allocateFilledTensor(M * OH * OW);
  RUN(convIm2col(InputCHW, FilterMCHW, im2colOutputMHW, C, H, W, M, FH, FW, OH, OW, PH, PW, SH, SW))
  std::cout << "Im2colCopy: " << Im2colCopy / Repeat << ", Im2colComp: " << Im2colComp / Repeat << "\n";
  Outputs.push_back(alignedAlloc(OH * OW * M));
  convertCHWToHWC(im2colOutputMHW, Outputs.back(), M, OH, OW);

  // Yaconv
  // Outputs.push_back(yaconvExtraOutput(H, W, FW, FW, PW, OW, M));
  // RUN(yaconv(InputHWC, FilterMHWC, Outputs.back(), C, H, W, M, FH, FW, PH, PW, SH, SW))
  // std::cout << "L3Pack: " << YaConvPack1 / Repeat << ", L2Pack: " << YaConvPack2 / Repeat << ", Comp: " << YaConvComp / Repeat << "\n";
  // clang-format on

  // Print tensors for each run
  // for (const auto &Output: Outputs)
  //   printTensor(Output, {OH * OW, M});

  // Print times for each run
  for (const auto &Time : Times)
    std::cout << Time << "\n";

  std::cout << "Max relative diff: " << maxRelativeDiff(Outputs, M * OH * OW) << "\n";

  // Free tensor memory
  free(InputCHW);
  free(InputHWC);
  free(FilterMCHW);
  free(FilterMHWC);
  free(im2colOutputMHW);
  for (const auto &Output: Outputs)
    free(Output);
  return 0;
}
