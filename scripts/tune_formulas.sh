#!/bin/bash
# ==============================================================================
# SMAC3 Hyperparameter Optimization Runner for Discovered Symbolic Formulas
# ==============================================================================
# This script tunes the continuous coefficients of the dynamic control policies:
#   1. e2e_policy        (Pure PySR skeleton, 10 continuous parameters)
#   2. co_design_policy  (Hybrid FOGA + PySR skeleton, 7 continuous parameters)
#
# Optimization is performed using Multi-Fidelity Hyperband / Successive Halving
# on OneMax problem instances with Common Random Numbers.
#
# By default, both policies ('e2e_policy' and 'co_design_policy') are tuned
# sequentially. You can also specify an individual policy as an argument.
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Help / Usage message
# ------------------------------------------------------------------------------
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat << 'EOF'
Usage: ./scripts/tune_formulas.sh [POLICIES...] [OPTIONS]

Tune continuous coefficients of symbolic policies using SMAC3 Multi-Fidelity.

Positional / Policy Arguments:
  e2e_policy | co_design_policy | all | both
                                Policy architecture(s) to optimize.
                                If omitted, defaults to tuning both policies sequentially.
                                (Aliases: e2e, pure_pysr, co_design, foga_pysr)

Configurable Options (passed to xdac/tune_formulas.py):
  -p, --policy POLICY           Target policy to tune.
  --combine PARAMS...           Active parameters to optimize: subset of [lbd_m alpha lbd_c beta].
                                (default: lbd_m alpha lbd_c beta)
  -n, --problem-sizes SIZES...  Problem sizes n for tuning evaluation.
                                (default: 100 200 500 1000 1500 2000)
  --total-budget BUDGET         Total budget in fidelity units (default: 1_000_000)
  --min-budget BUDGET           Minimum repetitions per trial (default: 50)
  --max-budget BUDGET           Maximum repetitions per trial (default: 1000)
  --eta FACTOR                  Hyperband successive halving factor (default: 3)
  --n-trials TRIALS             Explicit total trial limit (default: computed from budget)
  --total-cpus, --n-cpus CPUS   Total worker CPUs for parallelization (default: all available)
  --walltime-limit SECONDS      Optimization walltime limit in seconds (default: 30 days)
  --validation-reps REPS        Repetitions per size for final validation (default: max-budget)
  --validation-seed-offset SEED Starting seed for independent validation (default: 10000)
  -o, --out-dir DIR             Base output directory (default: outputs)
  --seed SEED                   Base random seed (default: 42)

Environment Variable Overrides:
  POLICY / POLICIES             Policy or policies to tune (e.g. "e2e_policy", "all")
  COMBINE                       Active controlled parameters (default: "lbd_m alpha lbd_c beta")
  PROBLEM_SIZES                 Problem sizes list (default: "100 200 500 1000 1500 2000")
  TOTAL_BUDGET                  Total optimization budget (default: 1000000)
  MIN_BUDGET                    Min reps per trial (default: 50)
  MAX_BUDGET                    Max reps per trial (default: 1000)
  ETA                           Successive halving factor (default: 3)
  N_CPUS                        Worker CPUs for SMAC (default: detected CPU count)
  WALLTIME_LIMIT                Walltime limit in seconds (default: 2592000)
  OUT_DIR                       Base output directory (default: outputs)
  SEED                          Master random seed (default: 42)
  PYTHON_BIN                    Python binary executable (default: python)

Examples:
  # 1. Tune both e2e_policy and co_design_policy sequentially with default settings
  ./scripts/tune_formulas.sh

  # 2. Tune e2e_policy only
  ./scripts/tune_formulas.sh e2e_policy

  # 3. Tune co_design_policy only using 32 worker CPUs
  ./scripts/tune_formulas.sh co_design_policy --total-cpus 32

  # 4. Ablation tuning: optimize only alpha, lbd_c, beta for e2e_policy
  ./scripts/tune_formulas.sh e2e_policy --combine alpha lbd_c beta

  # 5. Fast smoke/debugging run on smaller problem sizes with custom budget
  ./scripts/tune_formulas.sh e2e_policy -n 100 200 --min-budget 10 --max-budget 20 --n-trials 5

  # 6. Override via environment variables
  POLICY=co_design_policy N_CPUS=16 TOTAL_BUDGET=500000 ./scripts/tune_formulas.sh
EOF
    exit 0
fi

# ------------------------------------------------------------------------------
# Workspace and Environment Setup
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export WORKDIR="${WORKDIR:-$REPO_ROOT}"
cd "$WORKDIR"

# Load environment variables from .env if present
if [ -f .env ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
fi

# Add repository root to PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"

# Python executable check
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: Python executable '$PYTHON_BIN' not found. Please activate your environment." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# Parse Arguments and Identify Target Policies
# ------------------------------------------------------------------------------
TARGET_POLICIES=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--policy)
            TARGET_POLICIES+=("$2")
            shift 2
            ;;
        e2e_policy|co_design_policy|e2e|co_design|codesign|pure_pysr|foga_pysr)
            TARGET_POLICIES+=("$1")
            shift
            ;;
        all|both)
            TARGET_POLICIES=("e2e_policy" "co_design_policy")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Fallback to environment variables if no policy specified on CLI
if [ ${#TARGET_POLICIES[@]} -eq 0 ]; then
    if [ -n "$POLICY" ]; then
        if [ "$POLICY" = "all" ] || [ "$POLICY" = "both" ]; then
            TARGET_POLICIES=("e2e_policy" "co_design_policy")
        else
            # shellcheck disable=SC2206
            TARGET_POLICIES=($POLICY)
        fi
    elif [ -n "$POLICIES" ]; then
        if [ "$POLICIES" = "all" ] || [ "$POLICIES" = "both" ]; then
            TARGET_POLICIES=("e2e_policy" "co_design_policy")
        else
            # shellcheck disable=SC2206
            TARGET_POLICIES=($POLICIES)
        fi
    else
        # Default: Tune both policies
        TARGET_POLICIES=("e2e_policy" "co_design_policy")
    fi
fi

# ------------------------------------------------------------------------------
# Configurable Default Settings (Environment Variable Overrides)
# ------------------------------------------------------------------------------
COMBINE="${COMBINE:-lbd_m alpha lbd_c beta}"
PROBLEM_SIZES="${PROBLEM_SIZES:-100 200 500 1000 1500 2000}"
TOTAL_BUDGET="${TOTAL_BUDGET:-1000000}"
MIN_BUDGET="${MIN_BUDGET:-50}"
MAX_BUDGET="${MAX_BUDGET:-1000}"
ETA="${ETA:-3}"
N_CPUS="${N_CPUS:-$(nproc 2>/dev/null || echo 16)}"
WALLTIME_LIMIT="${WALLTIME_LIMIT:-2592000}"
OUT_DIR="${OUT_DIR:-outputs}"
SEED="${SEED:-42}"

echo "=========================================================="
echo "SMAC3 Formula Hyperparameter Optimization Pipeline"
echo "Working directory: $WORKDIR"
echo "Python binary:     $PYTHON_BIN"
echo "Policies to tune:  ${TARGET_POLICIES[*]}"
echo "Active parameters: $COMBINE"
echo "Problem sizes:     $PROBLEM_SIZES"
echo "Budgets:           min=$MIN_BUDGET, max=$MAX_BUDGET, total=$TOTAL_BUDGET, eta=$ETA"
echo "Worker CPUs:       $N_CPUS"
echo "Base Output dir:   $OUT_DIR"
echo "Random seed:       $SEED"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Additional CLI args: ${EXTRA_ARGS[*]}"
fi
echo "=========================================================="

# ------------------------------------------------------------------------------
# Run Tuning for each Target Policy
# ------------------------------------------------------------------------------
for POL in "${TARGET_POLICIES[@]}"; do
    echo ""
    echo "=========================================================="
    echo "Starting SMAC3 optimization for Policy: ${POL}..."
    echo "=========================================================="

    "$PYTHON_BIN" xdac/tune_formulas.py \
        --policy "$POL" \
        --combine $COMBINE \
        --problem-sizes $PROBLEM_SIZES \
        --total-budget "$TOTAL_BUDGET" \
        --min-budget "$MIN_BUDGET" \
        --max-budget "$MAX_BUDGET" \
        --eta "$ETA" \
        --total-cpus "$N_CPUS" \
        --walltime-limit "$WALLTIME_LIMIT" \
        --out-dir "$OUT_DIR" \
        --seed "$SEED" \
        "${EXTRA_ARGS[@]}"

    echo "Completed SMAC3 optimization for Policy: ${POL}"
done

echo ""
echo "=========================================================="
echo "All policy tuning tasks completed successfully."
echo "Artifacts and summaries saved under: $WORKDIR/$OUT_DIR"
echo "=========================================================="

