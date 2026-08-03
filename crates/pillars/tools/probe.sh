#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# 🔬 Probe — Benchmark & Profiling Toolkit for Pillars
# ═══════════════════════════════════════════════════════════════
#
# Usage:
#   ./tools/probe.sh [command]
#
# Commands:
#   bench              — run all criterion benchmarks
#   bench lif          — LIF benchmarks only
#   bench overhead     — overhead comparison benchmarks
#   perf               — perf stat on LIF benchmark binary
#   perf-record        — perf record + report
#   flamegraph         — generate flamegraph (requires perf + FlameGraph)
#   simd-check         — check which SIMD features are available
#   count              — count generated SIMD instructions in the binary
#
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/..")"

PROJECT="pillars"
CMD="${1:-bench}"
SUB="${2:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}═══ Pillars Probe ═══${NC}"
echo ""

case "$CMD" in

# ─── Benchmarks ─────────────────────────────────────────────
bench)
    if [ "$SUB" = "lif" ]; then
        echo -e "${YELLOW}Running LIF benchmarks...${NC}"
        cargo bench --package "$PROJECT" --bench bench_lif -- "$@"
    elif [ "$SUB" = "overhead" ]; then
        echo -e "${YELLOW}Running overhead comparison...${NC}"
        echo -e "${YELLOW}Compiling with tight loop for timing...${NC}"
        # Quick microbenchmark: compare raw vs pillars for LIF step
        cat > /tmp/pillars_overhead.rs << 'RUSTEOF'
// Quick overhead comparison: compile with `rustc /tmp/pillars_overhead.rs -o /tmp/probe`
// Uses inline assembly for cycle-counting
fn main() {
    use std::time::Instant;

    // Bare scalar: 138k LIF steps, no model trait
    let n = 138_000;
    let steps = 1000;
    let dt = 0.1f32;

    let mut v = vec![-70.0f32; n];
    let mut refr = vec![0.0f32; n];
    let i_in = vec![30.0f32; n];
    let tau_m = 10.0f32;
    let v_rest = -70.0f32;
    let v_thresh = -54.0f32;
    let v_reset = -70.0f32;
    let tau_ref = 2.0f32;

    // Raw scalar loop — no trait, no SIMD, just the math
    let start = Instant::now();
    for _ in 0..steps {
        for i in 0..n {
            let is_refr = refr[i] > 0.0;
            let dv = (v_rest - v[i] + i_in[i]) / tau_m * dt;
            let v_new = v[i] + dv;
            let v_dyn = if is_refr { v[i] } else { v_new };
            let spike = if !is_refr && v_dyn >= v_thresh { 1.0 } else { -1.0 };
            v[i] = if spike > 0.0 { v_reset } else { v_dyn };
            let refr_dec = refr[i] - dt;
            let refr_set = if spike > 0.0 { tau_ref } else { refr_dec };
            refr[i] = if refr_set > 0.0 { refr_set } else { 0.0 };
        }
    }
    let elapsed = start.elapsed();
    let ns_per_step = elapsed.as_nanos() as f64 / steps as f64;
    let ns_per_neuron = ns_per_step / n as f64;

    println!("SCALAR (branchy):");
    println!("  {:.2} ms total for {} steps × {} neurons", 
        elapsed.as_secs_f64() * 1000.0, steps, n);
    println!("  {:.1} ns/step, {:.3} ns/neuron", ns_per_step, ns_per_neuron);
    println!("  {} neurons/sec", (n as f64 / ns_per_neuron * 1e9) as u64);
}
RUSTEOF
        rustc -C opt-level=3 -C target-cpu=native /tmp/pillars_overhead.rs -o /tmp/pillars_probe 2>&1 && \
            /tmp/pillars_probe || echo "  (standalone probe needs pillars deps — run 'cargo bench' instead)"

        echo ""
        echo -e "${YELLOW}Use 'cargo bench --package pillars' for full benchmarks.${NC}"

    else
        echo -e "${YELLOW}Running all pillars benchmarks...${NC}"
        cargo bench --package "$PROJECT"
    fi
    ;;

# ─── Perf stat ──────────────────────────────────────────────
perf)
    echo -e "${YELLOW}Building benchmark binary...${NC}"
    cargo bench --package "$PROJECT" --bench bench_lif --no-run 2>&1 | tail -1
    BENCH_BIN=$(find target/release -maxdepth 3 -name "bench_lif*" -type f 2>/dev/null | head -1)
    if [ -z "$BENCH_BIN" ]; then
        echo -e "${RED}✗ Benchmark binary not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Benchmark: $BENCH_BIN${NC}"
    echo ""
    echo -e "${YELLOW}Running perf stat...${NC}"
    perf stat -e cycles,instructions,cache-misses,branch-misses,\
task-clock,context-switches,cpu-migrations,page-faults \
        "$BENCH_BIN" --bench 2>&1 | head -30
    ;;

# ─── Perf record + report ──────────────────────────────────
perf-record)
    echo -e "${YELLOW}Building benchmark binary...${NC}"
    cargo bench --package "$PROJECT" --bench bench_lif --no-run 2>&1 | tail -1
    BENCH_BIN=$(find target/release -maxdepth 3 -name "bench_lif*" -type f 2>/dev/null | head -1)
    if [ -z "$BENCH_BIN" ]; then
        echo -e "${RED}✗ Benchmark binary not found${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Running perf record...${NC}"
    perf record -F 1000 --call-graph dwarf -o /tmp/perf_pillars.data \
        "$BENCH_BIN" --bench 2>&1 | tail -5
    echo -e "${YELLOW}Generating report...${NC}"
    perf report -i /tmp/perf_pillars.data --stdio 2>&1 | head -60
    ;;

# ─── Flamegraph ─────────────────────────────────────────────
flamegraph)
    echo -e "${YELLOW}Generating flamegraph (requires perf + FlameGraph)...${NC}"
    echo -e "${YELLOW}Tip: Install FlameGraph from https://github.com/brendangregg/FlameGraph${NC}"
    echo ""
    echo "  perf record -F 1000 --call-graph dwarf -o /tmp/perf.data <bench_bin>"
    echo "  perf script -i /tmp/perf.data | stackcollapse-perf.pl | flamegraph.pl > flame.svg"
    ;;

# ─── SIMD feature check ─────────────────────────────────────
simd-check)
    echo -e "${YELLOW}Checking available SIMD features...${NC}"
    echo ""
    echo -e "  ${BOLD}CPU:${NC}"
    grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //'
    grep -m1 "flags" /proc/cpuinfo | grep -oP '(avx2|avx512f|avx512bw|fma|sse4_2|sse4_1|ssse3)' | sort -u
    echo ""
    echo -e "  ${BOLD}Rust target features:${NC}"
    rustc --print cfg -C target-cpu=native 2>/dev/null | grep -i -E '(avx|sse|fma|neon|sve)' | sed 's/^/    /'
    echo ""
    echo -e "  ${BOLD}Pillars SIMD lanes:${NC}"
    grep -r "pub const LANES" crates/pillars/src/arch.rs 2>/dev/null \
        || grep -r "LANES" crates/pillars/src/ --include="*.rs" | head -5
    ;;

# ─── SIMD instruction count ─────────────────────────────────
count)
    echo -e "${YELLOW}Building release and counting SIMD instructions...${NC}"
    cargo build --package "$PROJECT" --lib --release 2>&1 | tail -1
    LIB=$(find target/release -name "libpillars*" -type f 2>/dev/null | head -1)
    if [ -z "$LIB" ]; then
        echo -e "${RED}✗ Library not found${NC}"
        exit 1
    fi
    echo ""
    echo -e "${CYAN}SIMD instruction counts in libpillars.so:${NC}"
    objdump -d -M intel "$LIB" 2>/dev/null \
        | grep -oP '^\s+[0-9a-f]+:\s+([0-9a-f]{2}\s+)+\s+\K\w+' \
        | grep -iE '(vmovups|vmovdqu|vfmadd|vfnmadd|vfmsub|vmulps|vaddps|vsubps|vdivps|vblendvps|vxorps|vcmp|vbroadcast|vperm|vpshuf|vpmadd|vpsrl|vpand|vzeroupper|vmovmskps|vhaddps|vhsubps|vfmadd231|vfmadd132|vfmadd213)' \
        | sort | uniq -c | sort -rn | head -30
    echo ""
    TOTAL=$(objdump -d "$LIB" 2>/dev/null | grep -cP '^\s+[0-9a-f]+:\s+([0-9a-f]{2}\s+)+')
    SIMD=$(objdump -d "$LIB" 2>/dev/null | grep -cP '^\s+[0-9a-f]+:\s+([0-9a-f]{2}\s+)+\s+[av]')
    echo -e "${CYAN}Total instructions:${NC} $TOTAL"
    echo -e "${CYAN}SIMD instructions:${NC} $SIMD ($(( SIMD * 100 / TOTAL ))%)"
    ;;

*)
    echo -e "${YELLOW}Unknown command: $CMD${NC}"
    echo "Usage: $0 [bench|perf|perf-record|flamegraph|simd-check|count]"
    exit 1
    ;;
esac

echo ""
echo -e "${GREEN}Done.${NC}"
