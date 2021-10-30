#include "gemm.hpp"  // Tensor aligned allocation and printing
#include "utils.hpp"  // Tensor aligned allocation and printing
#include "yaconv.hpp" // class Indexer
#include <blis.h>     // BLIS microkernel and block sizes
#include <iostream>   // Debug printing
#include <set>        // Group construction

namespace {
#include "blis_params.hpp"
}; // namespace
void yaconv(float *Image, float *Filter, float *Output, int C,
            int H, int W, int M, int FH, int FW,
            int SH, int SW, int PH, int PW) {}
