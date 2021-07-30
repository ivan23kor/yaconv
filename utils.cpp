#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include "utils.h"



using namespace std;


float *allocateTensor(unsigned Size) { return new float[Size]; }

void randomizeTensor(float *&Tensor, unsigned Size) {
  srand(time(nullptr));
  for (unsigned i = 0; i < Size; ++i)
    Tensor[i] = rand() % 255;
}
 
void fillSerialTensor(float *&Tensor, unsigned Size) {
  for (unsigned i = 0; i < Size; ++i)
    Tensor[i] = i + 1;
}

void printVector(float *Vector, unsigned Len, const string Pre="", const string Post="\n") {
  cout << Pre << "[" << Vector[0];
  for (unsigned i = 1; i < Len; ++i)
    cout << setw(4) << Vector[i];
  cout << "]" << Post;
}

void printTensor(float *Tensor, vector<unsigned> Sizes, const string Pre, const string Post, bool First) {
  const string FirstPre = First ? "" : Pre;
  if (Sizes.size() == 1) {
    printVector(Tensor, Sizes[0], FirstPre, Post);
    return;
  }

  cout << FirstPre << "[";

  vector<unsigned> NewSizes(++Sizes.begin(), Sizes.end());
  if (Sizes[0] == 1) {
    printTensor(Tensor, NewSizes, "", "]" + Post);
    return;
  }

  printTensor(Tensor, NewSizes, Pre + " ", "\n");

  for (unsigned i = 0; i < NewSizes.size() - 1; ++i)
    cout << "\n";

  unsigned Stride = 1;
  for (const auto &s: NewSizes)
    Stride *= s;

  float *End = Tensor + (Sizes[0] - 1) * Stride;
  for (Tensor = Tensor + Stride; Tensor != End; Tensor += Stride)
    printTensor(Tensor, NewSizes, Pre + " ", "\n", false);
  printTensor(Tensor, NewSizes, Pre + " ", "]" + Post, false);
}
