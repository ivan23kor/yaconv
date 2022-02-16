CC := gcc
OPTFLAGS := -O2
CFLAGS :=

# If CHECK==1, check output correctness for yaconv
ifeq ($(CHECK),1)
CFLAGS += -DCHECK
endif

# Link flags
OPENBLAS_LINK := -lopenblas
BLIS_LINK := -lblis

# Binaries
OPENBLAS_TARGETS := im2col_openblas yaconv_openblas
BLIS_TARGETS := im2col_blis yaconv_blis
TARGETS := $(OPENBLAS_TARGETS) $(BLIS_TARGETS)

.PHONY: all clean

all : $(TARGETS)

openblas : $(OPENBLAS_TARGETS)

blis : $(BLIS_TARGETS)

im2col_openblas : conv.c
	$(CC) $(OPTFLAGS) $(CFLAGS) -DIM2COL -DOPENBLAS $< -o $@ $(OPENBLAS_LINK)

yaconv_openblas : conv.c
	$(CC) $(OPTFLAGS) $(CFLAGS) -DYACONV -DOPENBLAS $< -o $@ $(OPENBLAS_LINK)

im2col_blis : conv.c
	$(CC) $(OPTFLAGS) $(CFLAGS) -DIM2COL -DBLIS $< -o $@ $(BLIS_LINK)

yaconv_blis : conv.c
	$(CC) $(OPTFLAGS) $(CFLAGS) -DYACONV -DBLIS $< -o $@ $(BLIS_LINK)

clean :
	$(RM) $(TARGETS)
