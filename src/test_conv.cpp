#include "conv.hpp"
#include "yaconv.hpp"
#include "utils.hpp"
#include <blis.h>
#include <chrono>
#include <iostream>


using namespace std::chrono;
extern double YaConvPack1, YaConvPack2, YaConvComp;
extern double Im2colCopy, Im2colComp;


float *yaconvExtraOutput(int H, int W, int FH, int FW, int PW, int OW, int M) {
  auto *cntx = bli_gks_query_cntx();
  int NR = bli_cntx_get_blksz(BLIS_NR, cntx)->v[BLIS_FLOAT];
  int WholeH = H % NR == 0 ? H : H + NR - H % NR;
  int ExtraBefore = ((FH - 1) * OW + PW) * M;
  int OutputAndAfter = WholeH * OW * M + (W + PW - FW + 1) * M;

  auto *Ptr = alignedAlloc(ExtraBefore + OutputAndAfter);
  return Ptr + ExtraBefore;
}


int main(int argc, char **argv) {
  // Usage message
  if (argc < 12) {
    std::cerr << "Usage: ./test_conv N C H W M FH FW SH SW PH PW\n";
    return -1;
  }

  // CLI parsing
  const int N = atoi(argv[1]);
  const int C = atoi(argv[2]);
  const int H = atoi(argv[3]);
  const int W = atoi(argv[4]);
  const int M = atoi(argv[5]);
  const int FH = atoi(argv[6]);
  const int FW = atoi(argv[7]);
  const int PH = atoi(argv[8]);
  const int PW = atoi(argv[9]);
  const int SH = atoi(argv[10]);
  const int SW = atoi(argv[11]);

  // Input tensors
  auto *Input = allocateRandomTensor(N * C * H * W);

  // Filter tensors
  auto *Filter = allocateRandomTensor(M * C * FH * FW);

  // Output tensors
  const int OH = (H - FH + 2 * PH) / SH + 1;
  const int OW = (W - FW + 2 * PW) / SW + 1;
#ifdef YACONV // Allocate output for yaconv
  auto **Output = (float **)malloc(N * sizeof(float *));
  for (int n = 0; n < N; ++n)
    Output[n] = yaconvExtraOutput(H, W, FH, FW, PW, OW, M);
#else // Allocate output for im2col
  auto *Output = allocateRandomTensor(N * M * OH * OW);
#endif

  // Time measurement
  high_resolution_clock::time_point t1, t2;
  t1 = high_resolution_clock::now();

  for (int n = 0; n < N; ++n)
#ifdef YACONV // Convolution with yaconv
    yaconv(Input + n * C * H * W, Filter, Output[n], C, H, W, M, FH, FW, PH, PW, SH, SW);
#else // Convolution with im2col
    convIm2col(Input + n * C * H * W, Filter, Output + n * M * OH * OW, C, H, W, M, FH, FW, OH, OW, PH, PW, SH, SW);
#endif

  t2 = high_resolution_clock::now();
  double Time = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << Time << "\n";

  // Free tensor memory
  free(Input);
  free(Filter);
  // TODO: yaconv requires extra space
  // free(Output);

  return 0;
}
