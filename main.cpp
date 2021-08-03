#include <iostream>
#include "blis.h"
#include "im2col.h"
#include "mec.h"
#include "utils.h"


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
  cout << "===============================================================\n\n";

  // Init Input
  float *Input = allocateTensor(C_ * H_ * W_);
  fillSerialTensor(Input, C_ * H_ * W_);
  cout << "=== Input ===\n";
  printTensor(Input, {C_, H_, W_});
  cout << "===============================================================\n\n";

  // Calculate Output dimensions
  unsigned OH = H_ - KH_ + 1;
  unsigned OW = W_ - KW_ + 1;

  // im2col
  float *BufIm2col = allocateTensor(KH_ * KW_ * C_ * OH * OW);
  im2col_cpu(Input, C_, H_, W_, KH_, KW_, 0, 0, 1, 1, 1, 1, BufIm2col);
  cout << "=== BufIm2col ===\n";
  printTensor(BufIm2col, {C_ * KH_ * KW_, OH * OW});
  cout << "===============================================================\n\n";

  // Post-im2col GEMM
  float *OutputIm2col = allocateTensor(M_ * OH * OW);
  {
    float alpha = 1.0, beta = 0.0;
    unsigned M = M_;
    unsigned N = OH * OW;
    unsigned K = C_ * KH_ * KW_;
    unsigned rsa = C_ * KW_ * KW_;
    unsigned csa = 1;
    unsigned rsb = OH * OW;
    unsigned csb = 1;
    unsigned rsc = OH * OW;
    unsigned csc = 1;
    cout << M_ << " x " << K << " x " << N << "\n";
    cout << "[" << rsa << ", " << csa << "]\n";
    cout << "[" << rsb << ", " << csb << "]\n";
    cout << "[" << rsc << ", " << csc << "]\n";
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K,
        &alpha,
        Kernel, rsa, csa,
        BufIm2col, rsb, csb,
        &beta,
        OutputIm2col, rsc, csc
       );
    cout << "=== OutputIm2col ===\n";
    printTensor(OutputIm2col, {M_, OH * OW});
    cout << "===============================================================\n\n";
  }

  // MEC
  float *BufMec = mec(Input, C_, H_, W_, KH_, KW_, 0, 0, 1, 1, 1, 1);
  cout << "=== BufMec ===\n";
  printTensor(BufMec, {OW, H_ * KW_});
  cout << "===============================================================\n\n";

  // Post-mec GEMM
  float *OutputMec = allocateTensor(M_ * OH * OW);
  {
    float alpha = 1.0, beta = 0.0;
    unsigned M = OW;
    unsigned N = 1;
    unsigned K = KH_ * KW_;
    unsigned rsa = H_ * KW_;
    unsigned csa = 1;
    unsigned rsb = 1;
    unsigned csb = 1;
    unsigned rsc = 1;
    unsigned csc = 1;
    cout << M_ << " x " << K << " x " << N << "\n";
    cout << "[" << rsa << ", " << csa << "]\n";
    cout << "[" << rsb << ", " << csb << "]\n";
    cout << "[" << rsc << ", " << csc << "]\n";
    for (unsigned h = 0; h < OH; ++h) {
      bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K,
          &alpha,
          BufMec + h * KW_, rsa, csa,
          Kernel, rsb, csb,
          &beta,
          OutputMec + h * OW, rsc, csc
         );
    }
    cout << "=== OutputMec ===\n";
    printTensor(OutputMec, {M_, OH * OW});
    cout << "===============================================================\n\n";
  }
  cout << "im2col == MEC: " << tensorsEqual(OutputIm2col, OutputMec, M_ * OH * OW) << "\n";;

  delete[] Input;

  return 0;
} 

