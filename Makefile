CC := gcc
CXX := g++
FLAGS := -O3 -mavx2 -mfma -mfpmath=sse -march=haswell -funsafe-math-optimizations -ffp-contract=fast -fomit-frame-pointer

DEBUG ?= 0
DEBUG_FLAGS="-DDEBUG=$(DEBUG)"

OBJ := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

BLIS := -lblis
uKERNELS := bli_gemm_haswell_asm_d6x8.o

LIBS := $(BLIS)

yaconv: $(OBJ) $(uKERNELS)
	$(CXX) $(FLAGS) $^ -o $@ $(LIBS)

bli_gemm_haswell_asm_d6x8.o: /workdir/blis/kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c
	$(CC) $(FLAGS) -c $<

.PHONY: main.o
main.o: main.cpp
	$(CXX) $(DEBUG_FLAGS) $(FLAGS) $(BLOCKS) $(DIMS) -c $<

conv.o: conv.cpp config.h
	$(CXX) $(DEBUG_FLAGS) $(FLAGS) $(BLOCKS) -c $<

%.o: %.cpp
	$(CXX) $(FLAGS) -c $<

clean:
	$(RM) yaconv $(OBJ)
