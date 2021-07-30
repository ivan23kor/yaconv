#ifndef __UTILS_H_
#define __UTILS_H_


#include <string>
#include <vector>


float *allocateTensor(unsigned Size);

void randomizeTensor(float *&Tensor, unsigned Size);

void fillSerialTensor(float *&Tensor, unsigned Size);

void printTensor(float *Tensor, std::vector<unsigned> Sizes, const std::string Pre="", const std::string Post="\n", bool First=true);

#endif // __UTILS_H_
