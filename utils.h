#ifndef _UTILS_H_
#define _UTILS_H_

#include <string>
#include <vector>

float *allocateTensor(unsigned Size);

void randomizeTensor(float *&Tensor, unsigned Size, unsigned MaxVal=256);

void fillTensor(float *&Tensor, unsigned Size, float Value=-1.);

float *allocateFilledTensor(unsigned Size, float Value=-1.);

float *allocateRandomTensor(unsigned Size, unsigned MaxVal=256);

bool tensorsEqual(std::vector<float *>, const unsigned Size,
                  const float Epsilon = 1e-6);

void printTensor(float *Tensor, std::vector<unsigned> Sizes,
                 std::string Pre = "", const std::string Post = "\n",
                 bool First = true, int Setw=-1);

#endif // _UTILS_H_
