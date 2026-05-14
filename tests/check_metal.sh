#!/bin/bash
# check_metal.sh — compile all Metal sources into one library to verify syntax.
# Mimics what ds4_metal_full_source() does at runtime:
#   1. Concatenate preamble.metal + all .metal kernel files
#   2. Compile with xcrun (offline Metal compiler)
#   3. Report any errors

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
METAL_DIR="$SCRIPT_DIR/../metal"
TMPFILE=$(mktemp /tmp/l26f_metal_check_XXXXXX.metal)
METALLIB="${TMPFILE%.metal}.metallib"

SOURCES=(
    preamble.metal
    flash_attn.metal
    dense.metal
    moe.metal
    dsv4_hc.metal
    unary.metal
    dsv4_kv.metal
    dsv4_rope.metal
    dsv4_misc.metal
    argsort.metal
    cpy.metal
    concat.metal
    get_rows.metal
    repeat.metal
    sum_rows.metal
    softmax.metal
    glu.metal
    norm.metal
    bin.metal
    set_rows.metal
    l26f_dense.metal
    l26f_gla.metal
    l26f_matmul.metal
    l26f_mla.metal
    l26f_gdn.metal
    l26f_argmax.metal
    l26f_norm.metal
)

cleanup() { rm -f "$TMPFILE" "$METALLIB"; }
trap cleanup EXIT

> "$TMPFILE"
for src in "${SOURCES[@]}"; do
    path="$METAL_DIR/$src"
    if [ ! -f "$path" ]; then
        echo "MISSING: $path"
        exit 1
    fi
    echo "// --- $src ---" >> "$TMPFILE"
    cat "$path" >> "$TMPFILE"
    echo "" >> "$TMPFILE"
done

echo "Compiling $(wc -l < "$TMPFILE") lines of Metal source..."
if xcrun -sdk macosx metal "$TMPFILE" -o "$METALLIB" -std=metal3.1 2>&1; then
    echo "OK: all Metal sources compile successfully"
    # List kernel functions found
    KERNELS=$(xcrun metal-dump -dump-module "$METALLIB" 2>/dev/null | grep "kernel void" | wc -l || echo "?")
    echo "Kernel functions: $KERNELS"
else
    echo "FAILED: Metal compilation errors above"
    exit 1
fi
