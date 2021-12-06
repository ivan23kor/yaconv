# YaConv: Yet Another Convolution Algorithm

# Table of Contents
[Installation](#Installation)

# Installation

1. Install [blis](git@github.com:flame/blis.git)
1. Create symlinks to BLIS config files:
```bash
ln -s .../share/blis/config.mk config.mk
ln -s .../share/blis/config/ARCH/make_defs.mk make_defs.mk
```
1. Compile: `make all`
