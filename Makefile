CC := gcc
CXX := g++
FLAGS := -O3 -mavx2 -mfma -mfpmath=sse -march=haswell -funsafe-math-optimizations -ffp-contract=fast -fomit-frame-pointer

DEBUG ?= 0
DEBUG_FLAGS="-DDEBUG=$(DEBUG)"

OBJ := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

TARGETS := test_gemm test_conv

BLIS := -lblis
BLAS := -lblas
uKERNELS := bli_gemm_haswell_asm_d6x8.o

LIBS := $(BLIS) $(BLAS)

test_conv: gemm.o conv.o utils.o test_conv.o $(uKERNELS)
	$(CXX) $(FLAGS) $^ -o $@ $(LIBS)

test_gemm: gemm.o utils.o test_gemm.o $(uKERNELS)
	$(CXX) $(FLAGS) $^ -o $@ $(LIBS)

bli_gemm_haswell_asm_d6x8.o: /workdir/blis/kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c
	$(CC) $(FLAGS) -c $<

%.o: %.cpp
	$(CXX) $(FLAGS) $(DEBUG_FLAGS) -c $<

clean:
	$(RM) $(OBJ) $(TARGETS)
