#ifndef _UTILS_H_
#define _UTILS_H_

#include <string>
#include <vector>

float *allocateTensor(unsigned Size);

bool tensorsEqual(float *T1, float *T2, const unsigned Size,
                  const float Epsilon = 1e-6);

void randomizeTensor(float *&Tensor, unsigned Size);

void fillSerialTensor(float *&Tensor, unsigned Size);

void printTensor(float *Tensor, std::vector<unsigned> Sizes,
                 std::string Pre = "", const std::string Post = "\n",
                 bool First = true, int Setw=-1);

#endif // _UTILS_H_
