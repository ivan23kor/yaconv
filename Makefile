CXX := g++

include config.mk # blis/config.mk
include make_defs.mk # blis/config/ARCH/make_defs.mk

FLAGS := -std=c++11
FLAGS += $(CKOPTFLAGS) $(CKVECFLAGS)

BLIS := -lblis
LIBS := $(BLIS)

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
