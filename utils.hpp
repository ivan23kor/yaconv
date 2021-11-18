#pragma once

#include <string>
#include <vector>

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

void flushCache(int L3SizeInBytes=134217728);

float *alignedAlloc(int Size, int Alignment = 4096);

void randomizeTensor(float *&Tensor, int Size, int MaxVal = 256);

void fillTensor(float *&Tensor, int Size, float Value = -1.);

float *allocateFilledTensor(int Size, float Value = -1., int Alignment = -1);

float *allocateRandomTensor(int Size, int MaxVal = 256, int Alignment = -1);

float maxTensorDiff(std::vector<float *> Tensors, const int Size);

void printTensor(float *Tensor, std::vector<int> Sizes,
                 std::string Pre = "", const std::string Post = "\n",
                 bool First = true, int Setw = -1);
