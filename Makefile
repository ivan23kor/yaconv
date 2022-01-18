CC := gcc
OPTFLAGS := -O2

OPENBLAS_LINK := -lopenblas

TARGETS := im2col_openblas yaconv_openblas

.PHONY: all clean

all : $(TARGETS)

im2col_openblas : conv.c
	$(CC) $(OPTFLAGS) -DIM2COL_OPENBLAS $< -o $@ $(OPENBLAS_LINK)

yaconv_openblas : conv.c
	$(CC) $(OPTFLAGS) -DYACONV_OPENBLAS $< -o $@ $(OPENBLAS_LINK)

clean :
	$(RM) $(TARGETS)
