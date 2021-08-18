#include "conv.h"
#include "utils.h"
#include <blis.h>
#include <cblas.h>
#include <chrono>
#include <iostream>


using namespace std;
using namespace std::chrono;


#define MAIN_DEBUG(expr) if (DEBUG == 1) {expr;}


// Assumes strides = 1, pads = 0, dilations = 1
int main(int argc, char **argv) {
  if (argc != 8) {
    cerr << "Usage: ./yaconv N C H W M KH KW\n";
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
  MAIN_DEBUG(printTensor(Kernel, {M, C * KH * KW}))

  // Init Input
  float *Input = allocateFilledTensor(C * H * W);
  MAIN_DEBUG(printTensor(Input, {C, H, W}))

  // Time variables
  high_resolution_clock::time_point t1, t2;

  // Convolution with im2col
  float *OutputIm2col = allocateTensor(M * OH * OW);
  t1 = high_resolution_clock::now();
  convIm2col(Input, Kernel, OutputIm2col, C, H, W, M, KH, KW, OH, OW, 0,
              0, 1, 1, 1, 1);
  t2 = high_resolution_clock::now();
  double Im2colTime = duration_cast<duration<double>>(t2 - t1).count();

  // Convolution with fused im2col+packing
  float *OutputConvGemm = allocateTensor(M * OH * OW);
  t2 = high_resolution_clock::now();
  convGemm(Input, Kernel, OutputConvGemm, C, H, W, M, KH, KW);
  t2 = high_resolution_clock::now();
  double ConvGemmTime = duration_cast<duration<double>>(t2 - t1).count();

  cout << "Im2col: " << Im2colTime << "\n";
  cout << "ConvGemm: " << ConvGemmTime << "\n";

  MAIN_DEBUG(
    cout << "OutputIm2col:\n";
    printTensor(OutputIm2col, {M, OH * OW});
    cout << "OutputConvGemm:\n";
    printTensor(OutputConvGemm, {M, OH * OW});
  )

  return tensorsEqual(OutputIm2col, OutputConvGemm, M * OH * OW) ? 0 : -1;
}
