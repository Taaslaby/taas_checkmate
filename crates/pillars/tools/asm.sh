#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# 🔬 ASM Check — Disassembly Toolkit for Pillars
# ═══════════════════════════════════════════════════════════════
#
# Usage:
#   ./tools/asm.sh [model] [function]
#
# Examples:
#   ./tools/asm.sh LIFModel step        — LIF step() asm
#   ./tools/asm.sh HHModel step         — HH step() asm
#   ./tools/asm.sh run_neuron           — raw pointer runner asm
#   ./tools/asm.sh LIFModel all         — everything with LIFModel
#   ./tools/asm.sh                      — list available models
#
# Output: dumps annotated assembly with AVX2 instructions highlighted.
#
# Requirements: cargo, objdump (binutils)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/..")"

PROJECT="pillars"
TARGET_DIR="target/asm-dump"
MODEL="${1:-}"
FUNCTION="${2:-step}"

# ─── Colours ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Detect CPU ────────────────────────────────────────────
CPU=$(rustc --print target-cpus 2>/dev/null | grep "native" -A 0 | head -1 || echo "unknown")
echo -e "${CYAN}═══ Pillars ASM Check ═══${NC}"
echo -e "  CPU:      ${BOLD}${CPU}${NC}"
echo -e "  Target:   ${BOLD}x86_64-unknown-linux-gnu${NC}"
echo -e "  Model:    ${BOLD}${MODEL:-<all>}${NC}"
echo ""

# ─── No args → list models ─────────────────────────────────
if [ -z "$MODEL" ]; then
    echo -e "${YELLOW}Available models (pass as first arg):${NC}"
    grep -rn "impl Model for" crates/pillars/src/zoo/ --include="*.rs" \
        | sed 's/.*impl Model for //; s/ {$//' \
        | sort -u \
        | while read -r m; do echo "  • $m"; done
    echo ""
    echo -e "${YELLOW}Runner functions (pass 'run_neuron' or 'run_neuron_par'):${NC}"
    echo "  • run_neuron"
    echo "  • run_neuron_ext"
    echo "  • run_neuron_par (with rayon)"
    echo ""
    echo -e "Usage: ${BOLD}$0 <ModelName> [function]${NC}"
    echo "  function defaults to 'step'"
    echo "  Use 'all' to dump every function containing the model name"
    exit 0
fi

# ─── Build with debuginfo + assembly ───────────────────────
echo -e "${YELLOW}Compiling pillars with --emit asm...${NC}"
mkdir -p "$TARGET_DIR"

RUSTFLAGS="-C opt-level=3 -C target-cpu=native -C debuginfo=2" \
cargo build --package "$PROJECT" --lib --release \
    2>&1 | tail -3

# Find the compiled .rlib / .so
LIB_PATH=$(find target/release -name "libpillars*" -type f 2>/dev/null | head -1)
if [ -z "$LIB_PATH" ]; then
    echo -e "${RED}✗ Could not find compiled library${NC}"
    exit 1
fi
echo -e "  Library: ${BOLD}$LIB_PATH${NC}"

# ─── Generate assembly ─────────────────────────────────────
ASM_FILE="$TARGET_DIR/pillars.s"
RUSTFLAGS="-C opt-level=3 -C target-cpu=native -C llvm-args=-x86-asm-syntax=intel" \
cargo rustc --package "$PROJECT" --lib --release -- --emit asm -C "llvm-args=-x86-asm-syntax=intel" \
    2>/dev/null || true

# Try to find the generated .s file
ASM_SRC=$(find target/release/deps -name "*.s" -type f 2>/dev/null | head -1)
if [ -z "$ASM_SRC" ]; then
    ASM_SRC=$(find target -name "pillars*.s" -type f 2>/dev/null | head -1)
fi

# ─── Filter and display relevant assembly ─────────────────
if [ -n "$ASM_SRC" ]; then
    cp "$ASM_SRC" "$ASM_FILE"
    echo -e "${GREEN}✓ Assembly saved to: ${BOLD}$ASM_FILE${NC}"
    echo ""

    if [ "$FUNCTION" = "all" ]; then
        # Show everything containing the model name
        grep -A 500 "<.*$MODEL.*>:" "$ASM_FILE" | head -300
    else
        # Show specific function
        echo -e "${CYAN}═══ $MODEL :: $FUNCTION ═══${NC}"
        echo ""

        # Try matching the function name
        FUNC_PATTERN="<.*$MODEL.*$FUNCTION.*>:"
        if grep -q "$FUNC_PATTERN" "$ASM_FILE" 2>/dev/null; then
            grep -A 200 "$FUNC_PATTERN" "$ASM_FILE" \
                | head -200 \
                | sed "s/\(vmovups\|vmovdqu\|vfmadd\|vmulps\|vaddps\|vsubps\|vdivps\|vblendvps\|vxorps\|vcmp\|vbroadcast\|vperm\|vpshuf\|vpmadd\|vpsrl\|vpand\|vzeroupper\|vmovmskps\)/${GREEN}&${NC}/g"
        else
            echo -e "${YELLOW}Function not found directly in .s file. Trying objdump...${NC}"
        fi
    fi
fi

# ─── Also try objdump on the benchmark binary ────────────────
echo ""
echo -e "${CYAN}═══ objdump: $MODEL :: $FUNCTION ═══${NC}"
echo ""

# Target the compiled ELF binary (benchmark or example) — .rlib is an archive, not ELF
BENCH_BINARY=$(find target/release -name "bench_lif*" -type f 2>/dev/null | head -1)
if [ -z "$BENCH_BINARY" ]; then
    # Fallback: try building a minimal test binary
    BENCH_BINARY=$(find target/release -name "benchmark*" -type f 2>/dev/null | head -1)
fi
if [ -n "$BENCH_BINARY" ]; then
    objdump -d -M intel "$BENCH_BINARY" 2>/dev/null \
        | grep -A 100 "$MODEL.*$FUNCTION" \
        | head -150 \
        | sed "s/\(vmovups\|vmovdqu\|vfmadd\|vmulps\|vaddps\|vsubps\|vdivps\|vblendvps\|vxorps\|vcmp\|vbroadcast\|vperm\|vpshuf\|vpmadd\|vpsrl\|vpand\|vzeroupper\|vmovmskps\|vfnmadd\|vfmsub\|vhaddps\|vhsubps\)/${GREEN}&${NC}/g" \
        | sed "s/\(call\|ret\|jmp\|je\|jne\|jg\|jl\|jge\|jle\|ja\|jb\)/${YELLOW}&${NC}/g"
fi

# ─── Summary ───────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══ Instruction Count ═══${NC}"
if [ -n "$BENCH_BINARY" ]; then
    objdump -d -M intel "$BENCH_BINARY" 2>/dev/null \
        | grep -A 100 "$MODEL.*$FUNCTION" \
        | grep -oP '^\s+[0-9a-f]+:\s+([0-9a-f]{2}\s+)+\s+\K\w+' \
        | sort | uniq -c | sort -rn \
        | head -20
fi

echo ""
echo -e "${GREEN}Done.${NC}"
echo -e "${YELLOW}Tip:${NC} Full assembly at ${BOLD}$ASM_FILE${NC}"
echo -e "${YELLOW}Tip:${NC} To see specific SIMD instructions: grep -i 'vfmadd\\|vmovups' $ASM_FILE"
