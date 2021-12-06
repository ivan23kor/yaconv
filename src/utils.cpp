#include "utils.hpp"
#include <ctime>
#include <iomanip>
#include <iostream>

float *alignedAlloc(int Size, int Alignment) {
  float *Ans = nullptr;
  int Ret = posix_memalign((void **)&Ans, Alignment, Size * sizeof(float));

  if (Ret == 0)
    return Ans;

  // Handle bad alloc
  std::cerr << "\033[31m";
  if (Ret == ENOMEM)
    std::cerr << "Memory allocation error in posix_memalign" << std::endl;
  else if (Ret == EINVAL)
    std::cerr << "The alignment parameter in posix_memalign is not a power of 2 at least as large as sizeof(void *)" << std::endl;
  else
    std::cerr << "Unknown error in posix_memalign" << std::endl;
  std::cerr << "\033[0m";

  exit(Ret);
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

float maxRelativeDiff(std::vector<float *> Tensors, const int Size) {
  int N = Tensors.size();
  if (N < 2)
    return 0.0;

  float *TRef = Tensors[0];
  float MaxDiff = 0.0;
  for (int n = 1; n < N; ++n) {
    float *T = Tensors[n];
    for (int i = 0; i < Size; ++i)
      MaxDiff = MAX(MaxDiff, std::abs((TRef[i] - T[i]) / TRef[i]));
  }
  return MaxDiff;
}

void printVector(float *Vector, int Len, const std::string Pre = "",
                 const std::string Post = "\n", int Setw = 3) {
  std::cout << Pre << "[" << std::setprecision(10);
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
