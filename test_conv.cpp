#include "conv.hpp"
#include "utils.h"
#include <blis.h>
#include <cblas.h>
#include <chrono>
#include <iostream>


using namespace std::chrono;


#define MAIN_DEBUG(expr) if (DEBUG == 1) {expr;}


int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "Usage: ./test_conv N C H W M KH KW\n";
    return -1;
  }

  const unsigned N = atoi(argv[1]);
  const unsigned C = atoi(argv[2]);
  const unsigned H = atoi(argv[3]);
  const unsigned W = atoi(argv[4]);
  const unsigned M = atoi(argv[5]);
  const unsigned KH = atoi(argv[6]);
  const unsigned KW = atoi(argv[7]);
  const unsigned OH = H - KH + 1;
  const unsigned OW = W - KW + 1;

  // Init Kernel
  float *Kernel = allocateFilledTensor(M * C * KH * KW);
  MAIN_DEBUG(
    printTensor(Kernel, {M, C, KH * KW})
  )

  // Init Input
  float *Input = allocateFilledTensor(C * H * W);
  MAIN_DEBUG(
    printTensor(Input, {C, H, W})
  )

  // Time variables
  high_resolution_clock::time_point t1, t2;
  double TempTime;
  std::vector<double> Times;

  // Output tensors
  std::vector<float *> Outputs;

#define RUN(f) \
  Outputs.push_back(allocateTensor(M * OH * OW)); \
  TempTime = 0.0; \
  for (unsigned i = 0; i < 1; ++i) { \
    t1 = high_resolution_clock::now(); \
    f; \
    t2 = high_resolution_clock::now(); \
    TempTime += duration_cast<duration<double>>(t2 - t1).count(); \
  } \
  Times.push_back(TempTime);

  // Convolution with im2col
  RUN(convIm2col(Input, Kernel, Outputs.back(), C, H, W, M, KH, KW, OH, OW, 0, 0, 1, 1, 1, 1))

  // Convolution with MEC for NCHW format
  RUN(Outputs.back() = convMecNCHW(Input, Kernel, C, H, W, M, KH, KW))

  // // Convolution with fused im2col+packing
  RUN(Outputs.back() = convGemm(Input, Kernel, C, H, W, M, KH, KW))

  // Print times for each run
  for (const auto &Time: Times)
    std::cout << Time << "\n";

  // Print tensors for each run
  MAIN_DEBUG(
    for (const auto &Output: Outputs)
      printTensor(Output, {M, OH * OW});
  )

  return tensorsEqual(Outputs, M * OH * OW) ? 0 : -1;
}
