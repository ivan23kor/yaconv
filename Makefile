CC := gcc
CFLAGS := -O2

# If CHECK==1, check output correctness for yaconv
ifeq ($(CHECK),1)
CFLAGS += -DCHECK
endif

ROOT_DIR != dirname $(realpath $(firstword $(MAKEFILE_LIST)))

# BLIS configuration
BLIS_INSTALL_PATH := $(ROOT_DIR)/blis_install
BLIS_CONFIGURE_CMD := ./configure --prefix=$(BLIS_INSTALL_PATH) -a yaconv auto

# lib and include flags
CFLAGS += -I$(BLIS_INSTALL_PATH)/include
LDFLAGS += -L$(BLIS_INSTALL_PATH)/lib
LDLIBS += -l:libblis.a -lm -lpthread

# Target binaries
BINARIES := im2col yaconv

# Uppercase function
UC = $(shell echo '$1' | tr '[:lower:]' '[:upper:]')

.PHONY: all clean

all : blis_install $(BINARIES)

blis_install :
	cd blis && $(BLIS_CONFIGURE_CMD)
	$(MAKE) -C blis/ -j 8
	$(MAKE) -C blis/ install

% : conv.c
	$(CC) $(CFLAGS) -D$(call UC,$@) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean :
	$(RM) -r $(BINARIES) $(BLIS_INSTALL_PATH)
