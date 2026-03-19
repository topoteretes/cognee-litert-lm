#!/bin/bash
# Benchmark script for cognee-litert-lm Rust bindings on Android.
# Cross-compiles the simple_chat example, pushes it to an Android device,
# and runs knowledge-graph extraction benchmarks on CPU and GPU.
#
# Usage: ./benchmark_android.sh [--skip-build] [--skip-push]
#
# Environment variables:
#   MODEL_LOCAL       - Path to CPU/GPU model (default: ~/.litert-lm/models/gemma3-1b-it-int4.litertlm)
#   ANDROID_NDK_HOME  - Path to Android NDK (required for cross-compilation)

set -euo pipefail
exec 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="${ANDROID_HOME:-$HOME/Android/Sdk}/platform-tools:$PATH"

MODEL_LOCAL="${MODEL_LOCAL:-$HOME/.litert-lm/models/gemma3-1b-it-int4.litertlm}"
VENDOR_DIR="${SCRIPT_DIR}/vendor/LiteRT-LM"
PREBUILT_DIR="${VENDOR_DIR}/prebuilt/android_arm64"
DEVICE_DIR="/data/local/tmp"
MODEL_DEVICE="$DEVICE_DIR/model.litertlm"

SKIP_BUILD=false
SKIP_PUSH=false
for arg in "$@"; do
  case $arg in
    --skip-build) SKIP_BUILD=true ;;
    --skip-push) SKIP_PUSH=true ;;
  esac
done

# --- 1. Check device ---
echo "Checking device..."
DEVICE=$(adb devices | grep -w "device" | head -1 | cut -f1)
if [ -z "$DEVICE" ]; then
  echo "ERROR: No Android device found. Connect a device and retry."
  exit 1
fi
echo "Device: $DEVICE"
echo "Model:  $MODEL_LOCAL"

if [ ! -f "$MODEL_LOCAL" ]; then
  echo "ERROR: Model not found: $MODEL_LOCAL"
  exit 1
fi

# --- 2. Build ---
if [ "$SKIP_BUILD" = false ]; then
  # Determine the Android NDK linker.
  NDK_HOME="${ANDROID_NDK_HOME:-${NDK_HOME:-}}"
  if [ -z "$NDK_HOME" ]; then
    echo "ERROR: ANDROID_NDK_HOME is not set."
    exit 1
  fi

  TOOLCHAIN="$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64"
  export PATH="$TOOLCHAIN/bin:$PATH"

  # Point Cargo at the Bazel-produced shared library.
  export LITERT_LM_LIB_DIR="$VENDOR_DIR/bazel-bin/c"

  echo ""
  echo "Building C shared library (sync) via Bazel..."
  (cd "$VENDOR_DIR" && bazel build --config=android_arm64 //c:liblitert_lm_c.so 2>&1 | tail -5)
  cp -f "$LITERT_LM_LIB_DIR/liblitert_lm_c.so" /tmp/liblitert_lm_c_sync.so

  echo ""
  echo "Building C shared library (speculative) via Bazel..."
  (cd "$VENDOR_DIR" && bazel build --config=android_arm64 --define=async_constraint_masking=true //c:liblitert_lm_c.so 2>&1 | tail -5)
  cp -f "$LITERT_LM_LIB_DIR/liblitert_lm_c.so" /tmp/liblitert_lm_c_spec.so

  echo ""
  echo "Building simple_chat example for Android (aarch64)..."

  # Build without async-constraint-masking (sync).
  echo "  Building sync variant..."
  cargo build --example simple_chat \
    --features clap,serde_json \
    --target aarch64-linux-android \
    --release --no-default-features 2>&1 | tail -3
  cp -f target/aarch64-linux-android/release/examples/simple_chat /tmp/simple_chat_sync

  # Build with async-constraint-masking (speculative).
  echo "  Building speculative variant (async-constraint-masking)..."
  cargo build --example simple_chat \
    --features clap,serde_json,async-constraint-masking \
    --target aarch64-linux-android --release 2>&1 | tail -3
  cp -f target/aarch64-linux-android/release/examples/simple_chat /tmp/simple_chat_spec
else
  echo "Skipping build (--skip-build)"
fi

# --- 3. Push to device ---
if [ "$SKIP_PUSH" = false ]; then
  echo ""
  echo "Pushing binaries and libs to device..."

  adb shell "rm -f $DEVICE_DIR/simple_chat_sync $DEVICE_DIR/simple_chat_spec" 2>/dev/null || true

  adb push /tmp/simple_chat_sync "$DEVICE_DIR/simple_chat_sync"
  adb push /tmp/simple_chat_spec "$DEVICE_DIR/simple_chat_spec"
  adb shell "chmod +x $DEVICE_DIR/simple_chat_sync $DEVICE_DIR/simple_chat_spec"

  # Push both .so variants (sync and speculative).
  adb push /tmp/liblitert_lm_c_sync.so "$DEVICE_DIR/liblitert_lm_c_sync.so"
  adb push /tmp/liblitert_lm_c_spec.so "$DEVICE_DIR/liblitert_lm_c_spec.so"

  # Push GPU shared libs from prebuilt directory.
  if [ -d "$PREBUILT_DIR" ]; then
    for f in "$PREBUILT_DIR"/*.so; do
      [ -f "$f" ] && adb push "$f" "$DEVICE_DIR/"
    done
  fi

  # Push model if not already on device.
  if ! adb shell "test -f $MODEL_DEVICE" 2>/dev/null; then
    echo "Pushing model (this may take a moment)..."
    adb push "$MODEL_LOCAL" "$MODEL_DEVICE"
  else
    echo "Model already on device, skipping."
  fi
else
  echo "Skipping push (--skip-push)"
fi

# --- 4. Prepare knowledge extraction texts ---
KG_TEXT_1_HOST="/tmp/kg_text_1.txt"
KG_TEXT_2_HOST="/tmp/kg_text_2.txt"
KG_TEXT_3_HOST="/tmp/kg_text_3.txt"
KG_TEXT_4_HOST="/tmp/kg_text_4.txt"

cat > "$KG_TEXT_1_HOST" <<'EOF'
J. Robert Oppenheimer was the scientific director of the Manhattan Project's Los Alamos Laboratory.
He coordinated theoretical and experimental teams that designed and tested the first atomic bombs.
Oppenheimer worked with U.S. Army leadership and many physicists who had fled Europe.
EOF

cat > "$KG_TEXT_2_HOST" <<'EOF'
General Leslie Groves directed the Manhattan Engineer District for the U.S. Army Corps of Engineers.
Groves oversaw budget, logistics, security, and construction across major project sites.
He selected Oppenheimer to lead the scientific work at Los Alamos.
EOF

cat > "$KG_TEXT_3_HOST" <<'EOF'
Key Manhattan Project locations included Los Alamos in New Mexico, Oak Ridge in Tennessee, and Hanford in Washington.
Oak Ridge developed uranium enrichment processes, while Hanford produced plutonium.
The project integrated universities, government agencies, and industrial contractors.
EOF

cat > "$KG_TEXT_4_HOST" <<'EOF'
The Manhattan Project involved the U.S. Army Corps of Engineers, the Office of Scientific Research and Development,
and research groups from institutions such as the University of California and the University of Chicago.
Scientists Enrico Fermi, Niels Bohr, and Richard Feynman were associated with project efforts.
EOF

if [ "$SKIP_PUSH" = false ]; then
  adb push "$KG_TEXT_1_HOST" "$DEVICE_DIR/kg_text_1.txt"
  adb push "$KG_TEXT_2_HOST" "$DEVICE_DIR/kg_text_2.txt"
  adb push "$KG_TEXT_3_HOST" "$DEVICE_DIR/kg_text_3.txt"
  adb push "$KG_TEXT_4_HOST" "$DEVICE_DIR/kg_text_4.txt"
fi

# --- 5. Run benchmarks ---
echo ""
echo "================================================================"
echo "  BENCHMARK: Knowledge Graph Extraction (Rust Bindings)"
echo "  Device: $DEVICE"
echo "================================================================"

RESULT_LABELS=()
RESULT_PREFILL=()
RESULT_DECODE=()
RESULT_TOTAL=()

run_benchmark() {
  local label="$1"
  local run_id="$2"
  local text_file="$3"
  local binary="${4:-simple_chat_sync}"
  local backend="${5:-cpu}"
  local model="${6:-$MODEL_DEVICE}"
  local outfile="/tmp/bench_${run_id}.txt"

  echo ""
  echo "--- $label ---"
  echo "  Cooling down (5s)..."
  sleep 5

  adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./$binary \
    --model-path $model --backend $backend \
    --add-constraint \
    --input-text-file $text_file" \
    > "$outfile" 2>&1 || true

  local prefill decode total
  prefill=$(grep "Prefill Speed:" "$outfile" | head -1 | grep -oP '[\d.]+(?= tokens/sec)' || echo "N/A")
  decode=$(grep "Decode Speed:" "$outfile" | head -1 | grep -oP '[\d.]+(?= tokens/sec)' || echo "N/A")
  total=$(grep "Total Time:" "$outfile" | head -1 | grep -oP '[\d.]+(?= s)' || echo "N/A")

  echo "  Prefill: $prefill tok/s"
  echo "  Decode:  $decode tok/s"
  echo "  Total:   $total s"

  RESULT_LABELS+=("$label")
  RESULT_PREFILL+=("$prefill")
  RESULT_DECODE+=("$decode")
  RESULT_TOTAL+=("$total")
}

# --- CPU (uses sync .so) ---
echo ""
echo "Activating sync .so for CPU benchmarks..."
adb push /tmp/liblitert_lm_c_sync.so "$DEVICE_DIR/liblitert_lm_c.so"

run_benchmark "CPU text 1 (oppenheimer)"   "cpu_kg_text_1" "$DEVICE_DIR/kg_text_1.txt" simple_chat_sync cpu "$MODEL_DEVICE"
run_benchmark "CPU text 2 (groves)"        "cpu_kg_text_2" "$DEVICE_DIR/kg_text_2.txt" simple_chat_sync cpu "$MODEL_DEVICE"
run_benchmark "CPU text 3 (laboratories)"  "cpu_kg_text_3" "$DEVICE_DIR/kg_text_3.txt" simple_chat_sync cpu "$MODEL_DEVICE"
run_benchmark "CPU text 4 (organizations)" "cpu_kg_text_4" "$DEVICE_DIR/kg_text_4.txt" simple_chat_sync cpu "$MODEL_DEVICE"

# --- GPU sync (uses sync .so) ---
run_benchmark "GPU text 1 (oppenheimer)"   "gpu_kg_text_1" "$DEVICE_DIR/kg_text_1.txt" simple_chat_sync gpu "$MODEL_DEVICE"
run_benchmark "GPU text 2 (groves)"        "gpu_kg_text_2" "$DEVICE_DIR/kg_text_2.txt" simple_chat_sync gpu "$MODEL_DEVICE"
run_benchmark "GPU text 3 (laboratories)"  "gpu_kg_text_3" "$DEVICE_DIR/kg_text_3.txt" simple_chat_sync gpu "$MODEL_DEVICE"
run_benchmark "GPU text 4 (organizations)" "gpu_kg_text_4" "$DEVICE_DIR/kg_text_4.txt" simple_chat_sync gpu "$MODEL_DEVICE"

# --- GPU+spec (uses speculative .so with async constraint masking) ---
echo ""
echo "Activating speculative .so for GPU+spec benchmarks..."
adb push /tmp/liblitert_lm_c_spec.so "$DEVICE_DIR/liblitert_lm_c.so"

run_benchmark "GPU+spec text 1 (oppenheimer)"   "gpuspec_kg_text_1" "$DEVICE_DIR/kg_text_1.txt" simple_chat_spec gpu "$MODEL_DEVICE"
run_benchmark "GPU+spec text 2 (groves)"        "gpuspec_kg_text_2" "$DEVICE_DIR/kg_text_2.txt" simple_chat_spec gpu "$MODEL_DEVICE"
run_benchmark "GPU+spec text 3 (laboratories)"  "gpuspec_kg_text_3" "$DEVICE_DIR/kg_text_3.txt" simple_chat_spec gpu "$MODEL_DEVICE"
run_benchmark "GPU+spec text 4 (organizations)" "gpuspec_kg_text_4" "$DEVICE_DIR/kg_text_4.txt" simple_chat_spec gpu "$MODEL_DEVICE"

# --- Summary ---
echo ""
echo "================================================================"
echo "  SUMMARY TABLE"
echo "================================================================"
printf "| %-34s | %-14s | %-14s | %-10s |\n" "Run" "Prefill tok/s" "Decode tok/s" "Total (s)"
printf "|-%-34s-|-%-14s-|-%-14s-|-%-10s-|\n" "----------------------------------" "--------------" "--------------" "----------"
for i in "${!RESULT_LABELS[@]}"; do
  printf "| %-34s | %-14s | %-14s | %-10s |\n" \
    "${RESULT_LABELS[$i]}" "${RESULT_PREFILL[$i]}" "${RESULT_DECODE[$i]}" "${RESULT_TOTAL[$i]}"
done

echo ""
echo "================================================================"
echo "  DONE"
echo "================================================================"
