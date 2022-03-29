#!/bin/bash

[[ -n $1 ]] && BASE_DIR=$1 || BASE_DIR=Results

BINS="YACONV_BLIS.$HOSTNAME IM2COL_BLIS.$HOSTNAME"

REPEAT=10
MACHINE_GFLOPS=100


estimate_n_images() {
  local task_flops=$((2*$1*$2*$3*$4*$5*$6))
  echo $(($MACHINE_GFLOPS * 1000000000 / $task_flops))
}

run_layer() {
  local layer="${@:1:8}"
  local bin=$9

  local n_images=$(estimate_n_images ${layer[@]})
  local stat_file="$BASE_DIR/$bin/stat/$layer"
  local gflops_file="$BASE_DIR/$bin/gflops/$layer"

  echo -n "[$bin] $layer ... "

  perf stat -ddd -o "$stat_file" -x , -r $REPEAT \
    numactl -C0 ./$bin $n_images $layer > "$gflops_file"

  echo "Done!"
}

from_file() {
  local bin=$1
  local layers_file="./layers.txt"

  while read -a layer; do
    run_layer ${layer[@]} $bin
  done < $layers_file
}

grid() {
  local bin=$1

  for HW in 7 14 28 56; do
    for M in 32 64 96 128 192 256 384 512 768 960 1024 1152; do
      for C in 32 64 96 128 192 256 384 512 768 960 1024 1152; do
        run_layer $HW $HW $C 3 3 $M 1 1 $bin
      done
    done
  done
}

run() {
  local run_func=$1
  for bin in $BINS; do
    date
    mkdir -p $BASE_DIR/$bin/stat $BASE_DIR/$bin/gflops
    $run_func $bin
  done
  date
}

# run from_file
run grid
