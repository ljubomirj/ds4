#!/bin/bash
# Gold standard correctness test for ds4 REAP-compact
#
# Compares current generation output against recorded golden fixtures.
# All tests are deterministic (temp=0, fixed seed).
#
# Usage: ./tests/gold/run_test.sh
#   Returns 0 on pass, 1 on failure, prints details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GOLD_DIR="$SCRIPT_DIR"

MODEL="$HOME/ds4/gguf/DeepSeek-V4-Flash-REAP25-LCB50-DS4-compact-IQ2XXS.gguf"
BINARY="$SCRIPT_DIR/../../ds4"
PROMPT_FILE="$GOLD_DIR/prompt.txt"
GOLD_TEXT="$GOLD_DIR/expected_output.txt"
GOLD_LOGPROBS="$GOLD_DIR/expected_logprobs.json"
GOLD_TOKENS="$GOLD_DIR/expected_tokens.txt"
GOLD_SPEED="$GOLD_DIR/baseline_speed.txt"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
PASS=0
FAIL=0

check() {
    local name="$1"
    local status="$2"
    if [ "$status" -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $name"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC} $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== ds4 Gold Standard Test ==="
echo ""

# 1. Tokenization test
echo "1. Tokenization"
TOKENS_OUT=$( "$BINARY" -m "$MODEL" --prompt-file "$PROMPT_FILE" \
    --dump-tokens --backend metal 2>/dev/null | head -1 )
GOLD_TOKENS=$(cat "$GOLD_TOKENS" | tr -d '[] ')
TOKENS_CLEAN=$(echo "$TOKENS_OUT" | tr -d '[] ')
if [ "$TOKENS_CLEAN" = "$GOLD_TOKENS" ]; then
    check "Prompt tokenization matches" 0
else
    echo "     Expected: $GOLD_TOKENS"
    echo "     Got:      $TOKENS_CLEAN"
    check "Prompt tokenization matches" 1
fi

# 2. Output text test
echo "2. Output text"
TEXT_OUT=$( "$BINARY" -m "$MODEL" -p "$(cat "$PROMPT_FILE")" \
    -n 16 --temp 0 --seed 42 --backend metal 2>/dev/null )
TEXT_EXPECTED=$(cat "$GOLD_TEXT")
if [ "$TEXT_OUT" = "$TEXT_EXPECTED" ]; then
    check "Generated text matches" 0
else
    echo "     Expected: $TEXT_EXPECTED"
    echo "     Got:      $TEXT_OUT"
    check "Generated text matches" 1
fi

# 3. Logprobs structural test
echo "3. Logprobs (full precision comparison)"
python3 "$SCRIPT_DIR/compare_logprobs.py" \
    "$GOLD_LOGPROBS" \
    <( "$BINARY" -m "$MODEL" -p "$(cat "$PROMPT_FILE")" \
        -n 16 --temp 0 --seed 42 \
        --dump-logprobs /dev/stdout \
        --backend metal 2>/dev/null ) \
    && check "Logprobs match" 0 \
    || check "Logprobs match" 1

# 4. Speed sanity test
echo "4. Speed sanity"
SPEED_OUT=$( "$BINARY" -m "$MODEL" -p "$(cat "$PROMPT_FILE")" \
    -n 16 --temp 0 --seed 42 --backend metal 2>&1 >/dev/null | \
    grep -oE "generation: [0-9.]+ t/s" | grep -oE "[0-9.]+" )
echo "     Generation speed: $SPEED_OUT t/s (baseline: ~20-22 t/s)"
# Speed test is informational, not pass/fail — speed changes during optimization
echo "     (informational — not a pass/fail check)"
PASS=$((PASS + 1))  # count it as pass since it's informational

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
exit $FAIL
