#include "utils.hpp"
#include <ctime>
#include <iomanip>
#include <iostream>

float *alignedAlloc(int Size, int Alignment) {
  return (float *)aligned_alloc(Alignment, Size * sizeof(float));
}

void randomizeTensor(float *&Tensor, int Size, int MaxVal) {
  srand(time(nullptr));
  for (int i = 0; i < Size; ++i)
    Tensor[i] = rand() % MaxVal;
}

void fillTensor(float *&Tensor, int Size, float Value) {
  for (int i = 0; i < Size; ++i)
    Tensor[i] = Value < 0. ? i + 1 : Value;
}

float *allocateFilledTensor(int Size, float Value, int Alignment) {
  Alignment = Alignment > 0 ? Alignment : sizeof(void *);
  auto *Tensor = alignedAlloc(Size, Alignment);
  fillTensor(Tensor, Size, Value);
  return Tensor;
}

float *allocateRandomTensor(int Size, int MaxVal, int Alignment) {
  Alignment = Alignment > 0 ? Alignment : sizeof(void *);
  auto *Tensor = alignedAlloc(Size, Alignment);
  randomizeTensor(Tensor, Size, MaxVal);
  return Tensor;
}

bool tensorsEqual(std::vector<float *> Tensors, const int Size,
                  const float Epsilon) {
  int N = Tensors.size();
  if (N < 2)
    return true;

  float *TRef = Tensors[0];
  int Count = 0;
  for (int n = 1; n < N; ++n) {
    float *T = Tensors[n];
    for (int i = 0; i < Size; ++i) {
      float rel_diff = std::abs((TRef[i] - T[i]) / TRef[i]);
      if (rel_diff > Epsilon) {
        std::cerr << "[" << i << "] " << rel_diff << " |" << TRef[i] << " - "
                  << T[i] << "|\n";
        ++Count;
      }
    }
    if (Count > 0) {
      std::cout << Count << " values differ in Tensor[" << n << "].\n";
      break;
    }
  }

  return Count == 0;
}

void printVector(float *Vector, int Len, const std::string Pre = "",
                 const std::string Post = "\n", int Setw = 3) {
  std::cout << Pre << "[";
  for (int i = 0; i < Len; ++i)
    std::cout << std::setw(Setw + 1) << (Vector[i] > 0.1 ? Vector[i] : 0);
  std::cout << "]" << Post;
}

void printTensor(float *Tensor, std::vector<int> Sizes, std::string Pre,
                 const std::string Post, bool First, int Setw) {

  // This block only serves to determine setw width
  if (Setw == -1) {
    int Size = 1;
    for (const auto &s : Sizes)
      Size *= s;

    float MaxNum = 0.;
    for (int i = 0; i < Size; ++i)
      MaxNum = std::max(MaxNum, Tensor[i]);

    Setw = 1;
    while ((MaxNum /= 10) >= 1)
      ++Setw;
  }

  const std::string FirstPre = First ? "" : Pre;
  Pre += " ";
  if (Sizes.size() == 1) {
    printVector(Tensor, Sizes[0], FirstPre, Post, Setw);
    return;
  }

  std::cout << FirstPre << "[";

  std::vector<int> NewSizes(++Sizes.begin(), Sizes.end());
  if (Sizes[0] <= 1) {
    printTensor(Tensor, NewSizes, Pre, "]" + Post, true, Setw);
    return;
  }

  printTensor(Tensor, NewSizes, Pre, "\n", true, Setw);

  const std::string Newlines = std::string(NewSizes.size() - 1, '\n');
  std::cout << Newlines;

  int Stride = 1;
  for (const auto &s : NewSizes)
    Stride *= s;

  float *End = Tensor + (Sizes[0] - 1) * Stride;
  for (Tensor = Tensor + Stride; Tensor != End; Tensor += Stride) {
    printTensor(Tensor, NewSizes, Pre, "\n", false, Setw);
    std::cout << Newlines;
  }
  printTensor(Tensor, NewSizes, Pre, "]" + Post, false, Setw);
}

void __attribute__((optimize("O0"))) flushCache(int L3SizeInBytes) {
  int L3SizeInFloats = L3SizeInBytes / sizeof(float);
  auto *Dummy = allocateRandomTensor(L3SizeInFloats);
  free(Dummy);
}
