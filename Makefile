CC := gcc
CFLAGS := -O2

# If CHECK==1, check output correctness for yaconv
ifeq ($(CHECK),1)
CFLAGS += -DCHECK
endif

# Link flags
OPENBLAS_LINK := -lopenblas -lm
BLIS_LINK := -lblis -lm

# Set up targets
HOSTNAME := $(shell hostname)
ALGS := IM2COL YACONV
LIBS := BLIS OPENBLAS
OPENBLAS_TARGETS := $(foreach alg,$(ALGS),$(alg)_OPENBLAS.$(HOSTNAME))
BLIS_TARGETS := $(foreach alg,$(ALGS),$(alg)_BLIS.$(HOSTNAME))
TARGETS := $(BLIS_TARGETS)

# Parse alg and target names from target name
NAME_TUPLE = $(subst _, ,$(@:.$(HOSTNAME)=))
ALG = $(word 1,$(NAME_TUPLE))
LIB = $(word 2,$(NAME_TUPLE))

.PHONY: all blis openblas clean

all : $(TARGETS)

blis : $(BLIS_TARGETS)

openblas : $(OPENBLAS_TARGETS)

%.$(HOSTNAME) : conv.c
	$(CC) $(CFLAGS) -D$(ALG) -D$(LIB) $< -o $@ $($(LIB)_LINK)

clean :
	$(RM) $(TARGETS)
