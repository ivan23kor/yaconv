#include "blis.h"
#include "conv.h"
#include "utils.h"
#include <chrono>
#include <iostream>

using namespace std;

#ifndef C_
#define C_ 1
#endif

#ifndef H_
#define H_ 6
#endif

#ifndef W_
#define W_ 6
#endif

#ifndef M_
#define M_ 1
#endif

#ifndef KH_
#define KH_ 3
#endif

#ifndef KW_
#define KW_ 3
#endif

int main() {
  // Init Kernel
  float *Kernel = allocateTensor(M_ * C_ * KH_ * KW_);
  fillSerialTensor(Kernel, M_ * C_ * KH_ * KW_);
  cout << "=== Kernel ===\n";
  printTensor(Kernel, {M_, C_ * KH_ * KW_});
  cout << string(80, '-') << "\n\n";

  // Init Input
  float *Input = allocateTensor(C_ * H_ * W_);
  fillSerialTensor(Input, C_ * H_ * W_);
  cout << "=== Input ===\n";
  printTensor(Input, {C_, H_, W_});
  cout << string(80, '-') << "\n\n";

  // Calculate Output dimensions
  unsigned OH = H_ - KH_ + 1;
  unsigned OW = W_ - KW_ + 1;

  // Convolution with im2col
  float *OutputIm2col = allocateTensor(M_ * OH * OW);
  conv_im2col(Input, Kernel, OutputIm2col, C_, H_, W_, M_, KH_, KW_, OH, OW, 0,
              0, 1, 1, 1, 1);

  // Convolution with MEC
  float *OutputMec = allocateTensor(M_ * OH * OW);
  conv_mec(Input, Kernel, OutputMec, C_, H_, W_, M_, KH_, KW_, OH, OW, 0, 0, 1,
           1, 1, 1);

  // Check conv outputs match
  int ret = tensorsEqual(OutputIm2col, OutputMec, M_ * OH * OW) ? 0 : -1;

  delete[] Input;
  delete[] Kernel;
  delete[] OutputIm2col;
  delete[] OutputMec;

  return ret;
}
