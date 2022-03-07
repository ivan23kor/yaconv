#!/bin/bash


BASE_DIR=Results
BINS="YACONV_BLIS.$HOSTNAME IM2COL_BLIS.$HOSTNAME"

REPEAT=10
MACHINE_GFLOPS=100


estimate_n_images() {
  local task_gflops=$((2*$1*$2*$3*$4*$5*$6))
  echo $(($MACHINE_GFLOPS * 1000000000 / $task_gflops))
}


run_layer() {
  local layer=${@:1:8}
  local bin=$9

  local n_images=$(estimate_n_images ${layer[@]})
  local stat_file=$BASE_DIR/stat/$bin/${layer[@]}
  local gflops_file=$BASE_DIR/gflops/$bin/${layer[@]}

  echo -n "[$bin] ${layer[@]}..."

  perf stat -ddd -o "$stat_file" -x , -r $REPEAT \
    numactl -C0 ./$bin $n_images ${layer[@]} > "$gflops_file"

  echo "Done!"
}


run_from_file() {
  local bin=$1
  local layers_file=$2

  date
  while read -a layer; do
    run_layer ${layer[@]} $bin
  done < $layers_file
}


run_grid() {
  local bin=$1

  for M in {20..400..20}; do
    for C in {20..400..20}; do
      run_layer 20 20 $C 3 3 $M 1 1 $bin
    done
  done
}


for bin in $BINS; do
  mkdir -p $BASE_DIR/stat/$bin $BASE_DIR/gflops/$bin
  run_from_file $bin layers.txt
  #run_grid $bin
done
date
