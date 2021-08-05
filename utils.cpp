#include "utils.h"
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace std;

float *allocateTensor(unsigned Size) { return new float[Size]; }

void randomizeTensor(float *&Tensor, unsigned Size) {
  srand(time(nullptr));
  for (unsigned i = 0; i < Size; ++i)
    Tensor[i] = rand() % 255;
}

void fillTensor(float *&Tensor, unsigned Size, float Value) {
  for (unsigned i = 0; i < Size; ++i)
    Tensor[i] = Value < 0. ? i + 1 : Value;
}

float *allocateAndFillTensor(unsigned Size, float Value) {
  auto *Tensor = allocateTensor(Size);
  fillTensor(Tensor, Size, Value);
  return Tensor;
}

bool tensorsEqual(float *T1, float *T2, const unsigned Size,
                  const float Epsilon) {
  for (unsigned i = 0; i < Size; ++i)
    if (abs(T1[i] - T2[i]) > Epsilon)
      return false;
  return true;
}

void printVector(float *Vector, unsigned Len, const string Pre = "",
                 const string Post = "\n", int Setw=3) {
  cout << Pre << "[" << Vector[0];
  for (unsigned i = 1; i < Len; ++i)
    cout << setw(Setw + 1) << Vector[i];
  cout << "]" << Post;
}

void printTensor(float *Tensor, vector<unsigned> Sizes, string Pre,
                 const string Post, bool First, int Setw) {

  // This block only serves to determine setw width
  if (Setw == -1) {
    unsigned Size = 1;
    for (const auto &s : Sizes)
      Size *= s;

    float MaxNum = 0.;
    for (unsigned i = 0; i < Size; ++i)
      MaxNum = max(MaxNum, Tensor[i]);

    Setw = 1;
    while ((MaxNum /= 10.) > 1.)
      ++Setw;
  }

  const string FirstPre = First ? "" : Pre;
  Pre += " ";
  if (Sizes.size() == 1) {
    printVector(Tensor, Sizes[0], FirstPre, Post, Setw);
    return;
  }

  cout << FirstPre << "[";

  vector<unsigned> NewSizes(++Sizes.begin(), Sizes.end());
  if (Sizes[0] == 1) {
    printTensor(Tensor, NewSizes, Pre, "]" + Post, true, Setw);
    return;
  }

  printTensor(Tensor, NewSizes, Pre, "\n", true, Setw);

  const string Newlines = string(NewSizes.size() - 1, '\n');
  cout << Newlines;

  unsigned Stride = 1;
  for (const auto &s : NewSizes)
    Stride *= s;

  float *End = Tensor + (Sizes[0] - 1) * Stride;
  for (Tensor = Tensor + Stride; Tensor != End; Tensor += Stride) {
    printTensor(Tensor, NewSizes, Pre, "\n", false, Setw);
    cout << Newlines;
  }
  printTensor(Tensor, NewSizes, Pre, "]" + Post, false, Setw);
}
