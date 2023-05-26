#!/bin/bash

set -e

INPUT=grid
OUTPUT_DIR=Results
RUNS=10

usage() { echo "\
Usage: $0 -i=$INPUT ["grid"|"file"] -o=$OUTPUT_DIR <path> -r=$RUNS <number>

  -i=\"$INPUT\", \"grid\" or \"file\".
    When \"grid\", run on a hardcoded grid of parameters
    When \"file\", run on a list of layers given in layers.txt.
      Each layer should be on its own line, end with a newline and layer parameters are to be whitespace-separated, e.g.:
      ```
        7 7 1024 3 3 32 1 1
        5 5 128 3 3 32 1 1
        7 7 256 3 3 32 1 1

      ```
      Note the empty line at the end of the file.

  -o=\"$OUTPUT_DIR\"
    Path to where run results will be stored.
    Output directory structure:
      Results/
      └── IM2COL_BLIS.$HOSTNAME
          ├── gflops
          │   ├── layer1
          │   ├── layer2
          │   ├── ...
          └── stat
          │   ├── layer1
          │   ├── layer2
          │   ├── ...
      └── YACONV_BLIS.$HOSTNAME
          ├── gflops
          │   ├── layer1
          │   ├── layer2
          │   ├── ...
          └── stat
          │   ├── layer1
          │   ├── layer2
          │   ├── ...
    
    \"gflops\" files contain GFLOPS float numbers per run
    \"stat\" files contain averaged output of `perf stat`

  -r=$RUNS
    Number of averaging runs\
"; }


estimate_n_images() {
  WORKLOAD_GFLOPS=1
  local task_flops=$((2*$1*$2*$3*$4*$5*$6))
  echo $(($WORKLOAD_GFLOPS * 1000000000 / $task_flops))
}

run_layer() {
  local bin=$1
  local layer="${@:2}"

  local n_images=$(estimate_n_images ${layer[@]})
  local stat_file="$OUTPUT_DIR/$bin/stat/$layer"
  local gflops_file="$OUTPUT_DIR/$bin/gflops/$layer"

  echo -n "[$bin] $layer ... "

  sudo perf stat -ddd -o "$stat_file" -x , -r $RUNS \
    numactl -C 0 -m 0 ./$bin $n_images $layer > "$gflops_file"

  echo "Done!"
}

from_file() {
  local bin=$1
  local layers_file="./layers.txt"

  while read -a layer; do
    run_layer $bin ${layer[@]}
  done < $layers_file
}

grid() {
  local bin=$1

  for HW in 7 14 28 56; do
    for M in 32 64 96 128 192 256 384 512 768 960 1024 1152; do
      for C in 32 64 96 128 192 256 384 512 768 960 1024 1152; do
        run_layer $bin $HW $HW $C 3 3 $M 1 1
      done
    done
  done
}

run() {
  local run_func=$1
  BINS="YACONV_BLIS.$HOSTNAME IM2COL_BLIS.$HOSTNAME"
  for bin in $BINS; do
    ls $bin > /dev/null
    date
    mkdir -p $OUTPUT_DIR/$bin/stat $OUTPUT_DIR/$bin/gflops
    $run_func $bin
  done
  date
}

while getopts "i:o:r:h" flag; do
  case "$flag" in
    i)
      INPUT=$OPTARG
      [[ $INPUT != "grid" && $INPUT != "file" ]] && usage && exit 1
      ;;
    o)
      OUTPUT_DIR=$OPTARG
      [ -z "$OUTPUT_DIR" ] && usage && exit 1
      ;;
    r)
      RUNS=$OPTARG
      [ -z "$RUNS" ] && usage && exit 1
      ;;
    h) usage && exit 0;;
    *) usage && exit 1;;
  esac
done
shift "$((OPTIND-1))"

[[ $INPUT == "grid" ]] && run grid || run from_file
