#include "conv.hpp"
#include "yaconv.hpp"
#include "utils.hpp"
#include <blis.h>
#include <cblas.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;
extern double YaConvPack1, YaConvPack2, YaConvComp;

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
  int WholeH = H % 6 == 0 ? H : H + 6 - H % 6;
  int ExtraBefore = ((FH - 1) * OW + PW) * M;
  int OutputAndAfter = WholeH * OW * M + (W + PW - FW + 1) * M;
  // std::cout << ExtraBefore << " before and " << OutputAndAfter << " for output and after\n";

  auto *Ret = alignedAlloc(ExtraBefore + OutputAndAfter);
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
  auto *InputCHW = allocateFilledTensor(C * H * W);
  auto *InputHWC = allocateFilledTensor(H * W * C);
  convertCHWToHWC(InputCHW, InputHWC, C, H, W);

  // Create two filter tensors - MCHW and MHWC
  auto *FilterMCHW = allocateFilledTensor(M * C * FH * FW);
  auto *FilterMHWC = allocateFilledTensor(M * FH * FW * C);
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
  // Yaconv
  Outputs.push_back(yaconvExtraOutput(H, W, FW, FW, PW, OW, M));
  RUN(yaconv(InputHWC, FilterMHWC, Outputs.back(), C, H, W, M, FH, FW, PH, PW, SH, SW))
  std::cout << YaConvPack1 / Repeat << "\n";
  std::cout << YaConvPack2 / Repeat << "\n";
  std::cout << YaConvComp / Repeat << "\n";

  // Convolution with im2col
  auto *im2colOutputMHW = allocateFilledTensor(M * OH * OW);
  RUN(convIm2col(InputCHW, FilterMCHW, im2colOutputMHW, C, H, W, M, FH, FW, OH, OW, PH, PW, SH, SW))
  Outputs.push_back(alignedAlloc(OH * OW * M));
  convertCHWToHWC(im2colOutputMHW, Outputs[1], M, OH, OW);
  // clang-format on

  // Print tensors for each run
  // for (const auto &Output: Outputs)
  //   printTensor(Output, {OH * OW, M});

  // Print times for each run
  for (const auto &Time : Times)
    std::cout << Time << "\n";
  for (const auto &Time : Times)
    std::cout << 2 * M * C / 1000 * FH * FW * OH / 1000 * OW / Time / 1000 << "\n";
  // std::cout << Im2colTime / Repeat << "," << GEMMTime / Repeat << "\n";

  int Ret = tensorsEqual(Outputs, M * OH * OW) ? 0 : -1;

  // Free tensor memory
  free(InputCHW);
  free(InputHWC);
  free(FilterMCHW);
  free(FilterMHWC);
  free(im2colOutputMHW);
  // for (const auto &Output: Outputs)
  //   free(Output);
  return Ret;
}
