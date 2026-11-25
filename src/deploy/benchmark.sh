#!/usr/bin/env bash
# Usage: src/deploy/benchmark.sh <benchmark_model> <gpu|cpu> <model.tflite>...
#   gpu: S25, *_fp32.tflite (runs in FP16)   cpu: M02, *_int8.tflite
# Env: ANDROID_SERIAL (pick phone), THREADS=4, WARMUP=10, RUNS=100, DIR=/data/local/tmp/leafseg
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,5p "$0"; exit 1; }
BIN=$1 MODE=$2; shift 2
THREADS=${THREADS:-4} WARMUP=${WARMUP:-10} RUNS=${RUNS:-100} DIR=${DIR:-/data/local/tmp/leafseg}
case $MODE in
  gpu) FLAGS="--use_gpu=true --gpu_precision_loss_allowed=true" ;;
  cpu) FLAGS="--num_threads=$THREADS --use_xnnpack=true" ;;
  *) echo "mode must be gpu or cpu"; exit 1 ;;
esac

adb shell mkdir -p "$DIR"
adb push "$BIN" "$DIR/benchmark_model" > /dev/null
adb shell chmod +x "$DIR/benchmark_model"
echo "device: $(adb shell getprop ro.product.model | tr -d '\r')  mode: $MODE"
for M in "$@"; do
  adb push "$M" "$DIR/" > /dev/null
  echo "== $(basename "$M")"
  adb shell "$DIR/benchmark_model" --graph="$DIR/$(basename "$M")" $FLAGS \
      --warmup_runs="$WARMUP" --num_runs="$RUNS" 2>&1 | grep -E "Inference timings|Peak memory|ERROR|[Ff]ailed" || true
done
