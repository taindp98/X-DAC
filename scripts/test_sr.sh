#!/bin/bash
# ==============================================================================
# Scale-up and Ablation Testing Pipeline for Symbolic Regression Policies
# ==============================================================================
# This script evaluates PySR-discovered and handcrafted dynamic parameter control
# policies on the OneMax problem across various problem sizes to assess
# generalization, scalability, and parameter sensitivity (ablation analysis).
#
# Supported Policies:
#   - e2e_policy         (alias: e2e)
#       Pure end-to-end PySR symbolic regression expressions for all parameters.
#   - handcrafted_policy (aliases: handcrafted, foga)
#       Theoretical/handcrafted policy formulas derived in the FOGA paper.
#   - co_design_policy   (aliases: co_design, codesign)
#       Hybrid co-design policy marrying handcrafted formulas with PySR expressions.
#
# Controlled Parameters (for full control or ablation studies via --combine):
#   - lbd_m  (target 0): Mutation offspring population size lambda_m
#   - alpha  (target 1): Mutation scaling factor alpha
#   - lbd_c  (target 2): Crossover offspring population size lambda_c
#   - beta   (target 3): Crossover scaling factor beta
#   Note: When a parameter is omitted from --combine, it falls back to its predefined
#         default: lambda_m and lambda_c -> sqrt(n / (n - i)), alpha and beta -> 1.0.
#
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Help / Usage message
# ------------------------------------------------------------------------------
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat << 'EOF'
Usage: ./scripts/test_sr.sh [OPTIONS]

Run scale-up evaluation and ablation testing of symbolic policies on OneMax.
All arguments can be configured via environment variables or overridden by passing CLI flags.

Available Options (passed directly to xdac/test_sr.py):
  -p, --policy POLICY           Policy architecture to evaluate.
                                Choices: e2e_policy, handcrafted_policy, co_design_policy
                                Aliases: e2e, handcrafted, co_design, foga, codesign
                                (default: e2e_policy)

  --combine PARAMS...           Subset of parameters dynamically controlled by the policy.
                                Choices: lbd_m, alpha, lbd_c, beta
                                Omitted parameters revert to theoretical defaults.
                                (default: lbd_m alpha lbd_c beta)

  -n, --problem-sizes SIZES...  Problem sizes (n) to evaluate.
                                (default: 100 500 2000 3000 4000 5000 10000 20000 30000 40000)

  --n-reps, --reps REPS         Number of evaluation repetitions per problem size.
                                (default: 1000)

  --total-cpus, --n-cpus CPUS   Total CPU cores to utilize for parallel evaluation.
                                (default: 64)

  -o, --out-dir DIR             Base output directory for logging and results.
                                (default: outputs)

  --seed SEED                   Base random seed for reproducibility.
                                (default: 12345)

Environment Variable Overrides:
  POLICY                        Policy architecture (default: e2e_policy)
  COMBINE                       Controlled parameters (default: "lbd_m alpha lbd_c beta")
  PROBLEM_SIZES                 Problem sizes list (default: "100 500 2000 3000 4000 5000 10000 20000 30000 40000")
  N_REPS                        Evaluation repetitions (default: 1000)
  N_CPUS                        Number of parallel CPUs (default: 64)
  OUT_DIR                       Base output directory (default: outputs)
  SEED                          Random seed (default: 12345)
  PYTHON_BIN                    Python executable (default: python)

Examples:
  # 1. Run full evaluation of e2e_policy with default settings
  ./scripts/test_sr.sh

  # 2. Evaluate handcrafted (FOGA) policy
  ./scripts/test_sr.sh --policy handcrafted_policy

  # 3. Evaluate hybrid co-design policy
  ./scripts/test_sr.sh --policy co_design_policy

  # 4. Ablation study: evaluate policy controlling only alpha, lbd_c, beta (lbd_m disabled)
  ./scripts/test_sr.sh --policy e2e_policy --combine alpha lbd_c beta

  # 5. Fast smoke/debugging run on smaller problem sizes with fewer repetitions and CPUs
  ./scripts/test_sr.sh -n 100 500 --n-reps 20 --n-cpus 4

  # 6. Override via environment variables
  POLICY=co_design_policy N_REPS=500 N_CPUS=32 ./scripts/test_sr.sh
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

# Add the repository root to PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"

# Python executable check
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: Python executable '$PYTHON_BIN' not found. Please activate your conda/virtual environment." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# Configurable Settings (with Environment Variable Fallbacks)
# ------------------------------------------------------------------------------

# Policy architecture: 'e2e_policy', 'handcrafted_policy', or 'co_design_policy'
POLICY="${POLICY:-e2e_policy}"

# Controlled parameters (space-separated list):
# Can be any subset of: lbd_m alpha lbd_c beta
# If all 4 are provided, results are saved under default_results/.
# If any parameter is omitted, results are saved under ablation_results/.
COMBINE="${COMBINE:-lbd_m alpha lbd_c beta}"

# Problem dimensions (n) for generalization evaluation:
# Default benchmarks: 100, 500, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000
PROBLEM_SIZES="${PROBLEM_SIZES:-100 500 2000 3000 4000 5000 10000 20000 30000 40000}"

# Number of independent test repetitions per problem size
N_REPS="${N_REPS:-1000}"

# Number of parallel worker CPU cores (Joblib parallelization)
N_CPUS="${N_CPUS:-64}"

# Base output directory for logs and JSON evaluation results
OUT_DIR="${OUT_DIR:-outputs}"

# Master random seed for reproducibility
SEED="${SEED:-12345}"

# ------------------------------------------------------------------------------
# Run Evaluation
# ------------------------------------------------------------------------------
echo "=========================================================="
echo "OneMax PySR Symbolic Policy Evaluation Pipeline"
echo "Working directory: $WORKDIR"
echo "Python binary:     $PYTHON_BIN"
echo "Policy:            $POLICY"
echo "Active parameters: $COMBINE"
echo "Problem sizes:     $PROBLEM_SIZES"
echo "Repetitions:       $N_REPS"
echo "Worker CPUs:       $N_CPUS"
echo "Base Output dir:   $OUT_DIR"
echo "Random seed:       $SEED"
if [ $# -gt 0 ]; then
    echo "Additional CLI args: $*"
fi
echo "=========================================================="

"$PYTHON_BIN" xdac/test_sr.py \
    --policy "$POLICY" \
    --combine $COMBINE \
    --problem-sizes $PROBLEM_SIZES \
    --n-reps "$N_REPS" \
    --total-cpus "$N_CPUS" \
    --out-dir "$OUT_DIR" \
    --seed "$SEED" \
    "$@"

echo ""
echo "=========================================================="
echo "Evaluation task completed successfully."
echo "Results saved under: $WORKDIR/$OUT_DIR"
echo "=========================================================="

