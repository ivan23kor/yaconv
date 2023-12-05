
# YaConv: Convolution with Low Cache Footprint

## 1. Install dependencies

```bash
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r) numactl
```

## 2. Build

```bash
git clone --recurse-submodules https://github.com/ivan23kor/yaconv.git
cd yaconv && make
```

## 3. Run

1. Run either `im2col` or `yaconv` binary on a given layer:
```bash
./<BINARY> N H W C FH FW M PH PW  # outputs GFLOPS
```

| Parameter | Explanation                     |
| :-------: |:-------------------------------:|
| N         | number of images in per batch   |
| H, W      | image height and width          |
| C         | number of input channels        |
| FH, FW    | filter height and width         |
| M         | number of output channels       |
| PH, PW    | vertical and horizontal padding |

2. Gather comparative data for `im2col` and `yaconv` on a pre-defined grid:
```bash
./run.sh # writes run data on disk to ./Results/
```
The command above will run `./im2col` and `./yaconv` binaries on the following parameter grid:
| Parameter | Values
| :-------: |:---------------------------------------------:
| N         | Auto-adjusted to make ~100 GFLOPS workload per layer
| H, W      | 7  14  28  56
| C         | 32  64  96  128  192  256  384  512  768  960  1024  1152
| FH, FW    | 3
| M         | 32  64  96  128  192  256  384  512  768  960  1024  1152
| PH, PW    | 1

GFLOPS and `perf stat` data will be in a directory structure:
```
Results/
|-- im2col
|   |-- gflops
|   `-- stat
`-- yaconv
    |-- gflops
    `-- stat
```

For more information on the output in `./Results/`, run `./run.sh -h`

3. Gather comparative data for `im2col` or `yaconv` on a set of layers given in layers.txt:
```bash
./run.sh -i file  # (from layers.txt)
```

`./layers.txt` template:
```txt
7 7 1024 3 3 32 1 1
5 5 128 3 3 32 1 1
7 7 256 3 3 32 1 1

```
**Note the empty line at the end of the file.**

## Publication

You can find the paper [online](https://dl.acm.org/doi/10.1145/3570305) or download a [pdf](./YaConv.pdf) from this repository.
