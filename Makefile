CC := gcc
CXX := g++
CC_VENDOR := $(CC)

include make_defs.mk # blis/config/ARCH/make_defs.mk

FLAGS := -std=c++11
FLAGS += $(CKOPTFLAGS) $(CKVECFLAGS)

BLIS := -lblis
BLAS := -lblas
LIBS := $(BLIS) $(BLAS)

TARGETS := test_gemm test_conv

all: $(TARGETS)

test_gemm: gemm.o utils.o test_gemm.o
	$(CXX) $(FLAGS) $^ -o $@ $(LIBS)

test_conv: conv.o yaconv.o gemm.o utils.o test_conv.o
	$(CXX) $(FLAGS) $^ -o $@ $(LIBS)

%.o: %.cpp
	$(CXX) $(FLAGS) $(DEBUG_FLAGS) -c $<

clean:
	$(RM) *.o $(TARGETS)
