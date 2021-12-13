# https://github.com/TheNetAdmin/Makefile-Templates/blob/master/SmallProject/Template/Makefile

# BLIS configs contain compilation flags
include config.mk # share/blis/config.mk
include make_defs.mk # share/blis/config/ARCH/make_defs.mk

# tool macros
CXX ?= g++
override CXXFLAGS += -std=c++11
override CXXFLAGS += $(CKOPTFLAGS) $(CKVECFLAGS)

LIBS := -lblis

ifeq ($(OPENBLAS), 1)
	# use OpenBLAS gemm in im2col-based convolution (default: BLIS gemm) and
	# use OpenBLAS gemm in test_gemm alongside with BLIS
	LIBS += -lopenblas
	override CXXFLAGS += -DOPENBLAS_GEMM
endif

# path macros
OBJ_PATH := obj
SRC_PATH := src

# compile macros
TARGETS := conv gemm

# src files & obj files
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.cpp)))
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
GEMM_OBJ := $(filter-out %conv.o, $(OBJ))
CONV_OBJ := $(filter-out %gemm.o, $(OBJ))

# clean files list
CLEAN_LIST := $(TARGETS) $(OBJ)

# default rule
default: makedir all

gemm: $(GEMM_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

conv: $(CONV_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CXX) $(CXXFLAGS) $^ -c -o $@

# phony rules
.PHONY: makedir
makedir:
	@mkdir -p $(OBJ_PATH)

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)
