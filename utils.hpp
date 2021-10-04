#pragma once

#include <string>
#include <vector>

#define MIN(a, b) a < b ? a : b
 
// A macro to time algorithm parts
#ifndef TIME
#define TIME(cmd)                                                              \
  t1 = high_resolution_clock::now();                                           \
  cmd;                                                                         \
  t2 = high_resolution_clock::now();                                           \
  Times.push_back(duration_cast<duration<double>>(t2 - t1).count());
#else
#undef TIME
#define TIME(cmd) cmd;
#endif

// A macro to hide debug prints
#define IF_DEBUG(expr)                                                          \
  if (DEBUG == 1) {                                                             \
    expr;                                                                       \
  }

// A macro to allocate page-aligned output of SIZE floats, run and time some
// piece of code over REPEAT iterations, and to save the output and running
// time to the code-defined vectors `Outputs` and `Times`
#define RUN(SIZE, f, REPEAT)                                                   \
  Outputs.push_back(alignedAlloc(SIZE));                                       \
  TempTime = 0.0;                                                              \
  for (unsigned i = 0; i < REPEAT; ++i) {                                      \
    t1 = high_resolution_clock::now();                                         \
    f;                                                                         \
    t2 = high_resolution_clock::now();                                         \
    TempTime += duration_cast<duration<double>>(t2 - t1).count();              \
  }                                                                            \
  Times.push_back(TempTime / Repeat);

float *alignedAlloc(unsigned Size, int Alignment = 4096);

void randomizeTensor(float *&Tensor, unsigned Size, unsigned MaxVal = 256);

void fillTensor(float *&Tensor, unsigned Size, float Value = -1.);

float *allocateFilledTensor(unsigned Size, float Value = -1., int Alignment = -1);

float *allocateRandomTensor(unsigned Size, unsigned MaxVal = 256, int Alignment = -1);

bool tensorsEqual(std::vector<float *>, const unsigned Size,
                  const float Epsilon = 1e-6);

void printTensor(float *Tensor, std::vector<unsigned> Sizes,
                 std::string Pre = "", const std::string Post = "\n",
                 bool First = true, int Setw = -1);
