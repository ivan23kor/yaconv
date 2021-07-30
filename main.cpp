#include "im2col.h"
#include "utils.h"

#ifndef N
#define N 1
#endif

#ifndef C
#define C 1
#endif
  
#ifndef H
#define H 6
#endif

#ifndef W
#define W 6
#endif

#ifndef M
#define M 1
#endif
    
#ifndef K 
#define K 3
#endif

int main() {
  float *Input = allocateTensor(N * C * H * W);
  float *Kernel = allocateTensor(M * C * K * K);
  unsigned Ho = H - K + 1;
  unsigned Wo = W - K + 1;
  float *Output = allocateTensor(N * M * Ho * Wo);
  float *Patch = allocateTensor(K * K * N * C * Ho * Wo);
  
  fillSerialTensor(Input, N * C * H * W);
  printTensor(Input, {N, C, H, W});
  
  // im2col_cpu(Input, C, H, W, K, K, 0, 0, 1, 1, 1, 1, Patch);
    
  //printTensor(Patch, {Ho * Wo * K * K * C});

  return 0;
} 

