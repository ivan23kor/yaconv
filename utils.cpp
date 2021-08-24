#include "utils.h"
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace std;

float *allocateTensor(unsigned Size) { return new float[Size]; }

void randomizeTensor(float *&Tensor, unsigned Size, unsigned MaxVal) {
  srand(time(nullptr));
  for (unsigned i = 0; i < Size; ++i)
    Tensor[i] = rand() % MaxVal;
}

void fillTensor(float *&Tensor, unsigned Size, float Value) {
  for (unsigned i = 0; i < Size; ++i)
    Tensor[i] = Value < 0. ? i + 1 : Value;
}

float *allocateFilledTensor(unsigned Size, float Value) {
  auto *Tensor = allocateTensor(Size);
  fillTensor(Tensor, Size, Value);
  return Tensor;
}

float *allocateRandomTensor(unsigned Size, unsigned MaxVal) {
  auto *Tensor = allocateTensor(Size);
  randomizeTensor(Tensor, Size);
  return Tensor;
}

bool tensorsEqual(vector<float *>Tensors, const unsigned Size,
                  const float Epsilon) {
  unsigned N = Tensors.size();
  if (N < 2)
    return true;

  float *TRef = Tensors[0];
  unsigned Count = 0;
  for (unsigned n = 1; n < N; ++n) {
    float *T = Tensors[n];
    for (unsigned i = 0; i < Size; ++i) {
      float rel_diff = abs((TRef[i] - T[i]) / TRef[i]);
      if (rel_diff > Epsilon) {
        cerr << "[" << i << "] " << rel_diff << " |" << TRef[i] << " - " << T[i] << "|\n";
        ++Count;
      }
    }
    if (Count > 0) {
      cout << Count << " values differ in Tensor[" << n << "].\n";
      break;
    }
  }

  return Count == 0;
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
