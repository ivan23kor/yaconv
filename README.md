
# YaConv: Convolution with Low Cache Footprint

## Installation requirements

Linux toolchain (*GCC*, *perf*, *numactl*):
```bash
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r) numactl
```

## Build

```bash
git clone --recurse-submodules https://github.com/ivan23kor/yaconv.git
cd yaconv/blis
./configure -a yaconv auto
make -j 12
sudo make install
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib"
cd ../
make blis
```

## How to Run

1. Run an algorithm binary `BINARY` on a given layer:
```bash
./BINARY N H W C FH FW M PH PW  # outputs GFLOPS
```

| Parameter | Explanation                     |
| :-------: |:-------------------------------:|
| N         | number of images in per batch   |
| H, W      | image height and width          |
| C         | number of input channels        |
| FH, FW    | filter height and width         |
| M         | number of output channels       |
| PH, PW    | vertical and horizontal padding |

2. Gather comparative data for _IM2COL_ and _YACONV_ on a grid:
```bash
./run.sh
./run.sh -i grid  # (default)
```
The command above will run _IM2COL_BLIS_ and _YACONV_BLIS_ on the following parameter grid:
| Parameter | Values
| :-------: |:---------------------------------------------:
| N         | Auto-adjusted to make ~100 GFLOPS workload per layer
| H, W      | 7  14  28  56
| C         | 32  64  96  128  192  256  384  512  768  960  1024  1152
| FH, FW    | 3
| M         | 32  64  96  128  192  256  384  512  768  960  1024  1152
| PH, PW    | 1

3. Gather comparative data for _IM2COL_ and _YACONV_ on a set of layers given in a file:
```bash
./run.sh
./run.sh -i file  # (from layers.txt)
```
`layers.txt` should contain one layer per line. Layer parameters should be whitespace-separated, e.g.:
```txt
7 7 1024 3 3 32 1 1
5 5 128 3 3 32 1 1
7 7 256 3 3 32 1 1

```
**Note the empty line at the end of the file.**

The output will by default be stored in various files `./Results/`, more on that `./run.sh -h`