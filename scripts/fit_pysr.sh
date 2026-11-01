#!/bin/bash
# Symbolic Regression Fitting for OneMax Factored DDQN Policies using PySR
# Matches hyperparameters and targets specified in symbolic_regression/sr_fitting.txt

set -e

# Help / Usage message
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [TARGETS...]"
    echo ""
    echo "Run Symbolic Regression fitting using PySR on OneMax Factored DDQN policies."
    echo ""
    echo "Arguments:"
    echo "  TARGETS              Target indices to fit (e.g. 0, 1, 2, 3, or 'all')."
    echo "                       If omitted, defaults to fitting all 4 targets (0 1 2 3) sequentially."
    echo ""
    echo "Environment Variables (Overrides):"
    echo "  NCPUS                Number of worker CPUs (default: 12)"
    echo "  NITERATIONS          PySR iterations (default: 5000)"
    echo "  MAXSIZE              Max equation complexity (default: 12)"
    echo "  PARSIMONY            Parsimony penalty coefficient (default: 0.0001)"
    echo "  POPULATION_SIZE      Population size per island (default: 50)"
    echo "  LOSS                 Loss function: 'mse' or 'relative' (default: mse)"
    echo "  OUT_DIR              Output base directory (default: outputs)"
    echo "  PYTHON_BIN           Path to python binary (default: python)"
    echo ""
    echo "Examples:"
    echo "  $0                   # Fit targets 0, 1, 2, 3 sequentially"
    echo "  $0 0                 # Fit target 0 only"
    echo "  $0 0 1               # Fit targets 0 and 1"
    echo "  NCPUS=8 $0 0         # Fit target 0 using 8 CPUs"
    exit 0
fi

# Set working directory to the root tree folder of OneMax-MPDAC
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export WORKDIR="${WORKDIR:-$REPO_ROOT}"
cd "$WORKDIR"

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Add the repository root to PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"

# Python executable
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: Python executable '$PYTHON_BIN' not found. Please activate the conda environment." >&2
    exit 1
fi

# Ensure writable Julia project for juliapkg / PySR
if [ -z "$PYTHON_JULIAPKG_PROJECT" ]; then
    if [ -n "$SCRATCH" ] && [ -d "$SCRATCH/.julia_env" ]; then
        export PYTHON_JULIAPKG_PROJECT="$SCRATCH/.julia_env"
    elif [ -d "$HOME/.julia_env" ]; then
        export PYTHON_JULIAPKG_PROJECT="$HOME/.julia_env"
    else
        export PYTHON_JULIAPKG_PROJECT="$HOME/.julia_env"
    fi
fi

# Synchronize timestamp across all targets in this run
export SR_DATE="${SR_DATE:-$(date +%Y-%m-%d)}"
export SR_TIME="${SR_TIME:-$(date +%H-%M-%S)}"

# Hyperparameters matching symbolic_regression/sr_fitting.txt
NCPUS="${NCPUS:-12}"
NITERATIONS="${NITERATIONS:-5000}"
MAXSIZE="${MAXSIZE:-12}"
PARSIMONY="${PARSIMONY:-0.0001}"
POPULATION_SIZE="${POPULATION_SIZE:-50}"
LOSS="${LOSS:-mse}"
OUT_DIR="${OUT_DIR:-outputs}"

# Parse targets from command line arguments, or default to all 4 targets (0, 1, 2, 3)
if [ $# -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=(0 1 2 3)
fi

echo "=========================================================="
echo "OneMax Symbolic Regression Fitting Pipeline"
echo "Working directory: $WORKDIR"
echo "Python binary:     $PYTHON_BIN"
echo "Output directory:  $WORKDIR/$OUT_DIR/$SR_DATE/$SR_TIME"
echo "Targets to fit:    ${TARGETS[*]}"
echo "Hyperparameters:   ncpus=$NCPUS, niterations=$NITERATIONS, maxsize=$MAXSIZE, parsimony=$PARSIMONY, pop_size=$POPULATION_SIZE, loss=$LOSS"
echo "=========================================================="

for TARGET in "${TARGETS[@]}"; do
    echo ""
    echo "=========================================================="
    echo "Starting PySR fitting for Target ${TARGET}..."
    echo "=========================================================="
    "$PYTHON_BIN" xdac/fit_pysr.py \
        --target "$TARGET" \
        --ncpus "$NCPUS" \
        --niterations "$NITERATIONS" \
        --maxsize "$MAXSIZE" \
        --parsimony "$PARSIMONY" \
        --population-size "$POPULATION_SIZE" \
        --loss "$LOSS" \
        --out-dir "$OUT_DIR"
done

echo ""
echo "=========================================================="
echo "All PySR fitting tasks completed successfully."
echo "Artifacts saved at: $WORKDIR/$OUT_DIR/$SR_DATE/$SR_TIME"
echo "=========================================================="

