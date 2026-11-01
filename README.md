<h1>
    <p align="center"> Discovering Interpretable Multi-Parameter Control Policies for Evolutionary Algorithms Using Deep Reinforcement Learning </p>
</h1>

## 🗒️ Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [1. Conda Environment & Core Dependencies](#1-conda-environment--core-dependencies)
  - [2. Symbolic Regression (PySR) Backend Setup](#2-symbolic-regression-pysr-backend-setup)
- [Quickstart](#quickstart)
  - [Testing](#testing)
  - [Baseline and Comparison](#baseline-and-comparison)
  - [Training](#training)
- [Symbolic Policy Discovery](#symbolic-policy-discovery)
  - [Stage I: Behavioral Distillation via Symbolic Regression (PySR)](#stage-i-behavioral-distillation-via-symbolic-regression-pysr)
  - [Stage II: Online Environmental Calibration (SMAC3)](#stage-ii-online-environmental-calibration-smac3)

## 💡 Introduction
![](./assets/xdac.png)


This repository accompanies the paper *"Discovering Interpretable Multi-Parameter Control Policies for Evolutionary Algorithms Using Deep Reinforcement Learning"* (IEEE Transactions on Evolutionary Computation). We study dynamic control of all four parameters $(\lambda_m, \alpha, \lambda_c, \beta)$ of the $(1+(\lambda,\lambda))$-GA on OneMax using deep-RL, and distill the learned behaviors into interpretable symbolic policies. Key contributions:

- **RL enhancements:** factored action-space decomposition, adaptive reward shifting, and long-horizon discounting together enable effective multi-parameter learning.
- **DDQN vs. PPO:** PPO collapses to a degenerate policy in this DAC setting; DDQN with the above enhancements consistently learns high-quality policies.
- **Symbolic policy discovery:** a two-stage pipeline (hand-crafted equations → SMAC3 fine-tuning) yields a policy that outperforms all baselines, including IRACE, across problem sizes up to $n = 40{,}000$.

The environment extends [DACBench](https://github.com/automl/DACBench). See [dacbench/](dacbench/) for details.

## 🎯 Repository Structure

Outline the structure of repository.

```plaintext
OneMax-MPDAC/
├── assets/                             # Figures used in this README
│   └── xdac.png                        # Interpretable RL for Dynamic Algorithm Configuration framework (Fig. 1 in paper)
├── notebooks/
│   └── test.ipynb                      # Interactive notebook: load a checkpoint and observe ERT
├── resources/
│   ├── ddqn_ckpts/                     # Best DDQN checkpoints (factored, AS, γ=0.9998)
│   │   └── onemax_n{100,200,500,1000,1500,2000}_fmp_as_09998.pt
│   └── runtimes/
│       └── comparison_method_runtimes.json  # Pre-computed ERTs for all baselines
├── dacbench/                           # Extended DACBench environment
│   ├── benchmarks/
│   │   └── theory_benchmark.py         # Benchmark wrapper for the (1+(λ,λ))-GA
│   ├── envs/
│   │   ├── theory.py                   # Full 4-parameter (1+(λ,λ))-GA environment
│   │   ├── ablation_l1_theory.py       # Ablation: control λm only
│   │   ├── ablation_l1c_theory.py      # Ablation: control λm + α
│   │   ├── ablation_l1l2_theory.py     # Ablation: control λm + λc
│   │   ├── ablation_l1l2c_theory.py    # Ablation: control λm + α + λc
│   │   ├── ablation_l1l2m_theory.py    # Ablation: control λm + λc + β
│   │   ├── ablation_l1m_theory.py      # Ablation: control λm + β
│   │   ├── ablation_l1mc_theory.py     # Ablation: control λm + α + β
│   │   ├── ablation_ppo.py             # Environment variant used for PPO experiments
│   │   └── policies/theory/            # Theory-derived policy (πTHEORY) implementation
│   ├── instance_sets/
│   │   └── ollga_theory/               # Problem instance CSVs for n ∈ {50…40000}
│   ├── run_baselines.py                # Script to run and record baseline policy runtimes
│   └── wrappers/                       # Gym wrappers (action tracking, observation, reward noise, …)
├── xdac/                       # Main project source
│   ├── comb_ddqn.py                    # Combinatorial DDQN (2401-output Q-network)
│   ├── fact_ddqn.py                    # Factored DDQN (4-branch action-branching network)
│   ├── train_ddqn.py                   # Training script for DDQN (combinatorial & factored)
│   ├── train_comb_ppo.py               # Training script for PPO with combinatorial action space
│   ├── train_fact_ppo.py               # Training script for PPO with factored action space
│   ├── fit_pysr.py                     # Stage I: Symbolic regression policy distillation via PySR
│   ├── eval.py                         # Evaluation utilities (run policy, compute ERT)
│   ├── tune_derive_mp_policy.py        # Stage II: fine-tune symbolic policy via SMAC3
│   ├── loggers.py                      # TensorBoard / WandB logging helpers
│   ├── utils.py                        # Shared utilities
│   └── configs/
│       ├── onemax_n{N}_cmp[_as].yml    # DDQN combinatorial configs (N ∈ {100…2000}, optional AS)
│       ├── onemax_n{N}_fmp[_as][_l1*].yml  # DDQN factored configs (with ablation variants)
│       ├── onemax_n{N}_sp_l1[_as].yml  # Single-parameter DDQN configs
│       ├── onemax_n{N}_mp_ppo[_l1*].yml    # Multi-parameter PPO configs (factored & ablation)
│       ├── onemax_n{N}_sp_ppo.yml      # Single-parameter PPO configs
│       ├── ablation_ppo/               # PPO reward-shifting ablation configs (b ∈ {0,−1,…,−7})
│       ├── search_space/               # SMAC3 hyperparameter search space definitions
│       └── target_function/            # SMAC3 target algorithm wrapper configs
├── scripts/
│   ├── fit_pysr.sh                     # Runner script for Stage I PySR fitting
│   └── run.sh                          # Convenience shell script for launching experiments
├── requirements.txt
├── README.md
└── LICENSE
```

**Config naming conventions:**
- `cmp` / `fmp` — combinatorial / factored action space representation
- `sp` / `mp` — single-parameter / multi-parameter control
- `as` — adaptive reward shifting enabled
- `l1`, `l1c`, `l1l2`, `l1l2c`, `l1l2m`, `l1m`, `l1mc` — parameter subsets used in ablation study (Section VI of paper): `l1`=λm, `l2`=λc, `c`=α, `m`=β

## ⚙️ Installation

To reproduce this project, you will need the following dependencies:
- Linux (Ubuntu 18.04 LTS or later)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.10
- [PyTorch](https://pytorch.org/) (version 2.0 or later)

### 1. Conda Environment & Core Dependencies

Create and activate a new Conda environment:

```bash
conda create -n onemaxmpdac python=3.10
conda activate onemaxmpdac
```

Install PyTorch according to your hardware/CUDA version (refer to [Installing Previous Versions of PyTorch](https://pytorch.org/get-started/previous-versions/)):

```bash
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Symbolic Regression (PySR) Backend Setup

Stage I policy distillation uses **PySR** (Symbolic Regression via Genetic Programming), which uses Julia (`SymbolicRegression.jl`) as its high-performance backend via `juliapkg` / `juliacall`.

#### Julia Backend Configuration

`juliapkg` automatically handles downloading and managing the compatible Julia binary upon first initialization.

> [!IMPORTANT]
> In shared HPC clusters or multi-user environments where Conda/Python is located in a read-only or shared directory, `juliapkg` may attempt to write to system directories and fail with permission errors. To ensure an isolated, fully user-writable workspace for Julia packages and precompiled system images, set the `PYTHON_JULIAPKG_PROJECT` environment variable:

```bash
# Add to your ~/.bashrc or export in your active session:
export PYTHON_JULIAPKG_PROJECT="$HOME/.julia_env"
```

#### Precompile the Backend

Initialize PySR once to download the Julia runtime, install `SymbolicRegression.jl`, and precompile all Julia dependencies:

```bash
python -c "import pysr"
```

*Note: Initial precompilation takes approximately 1–2 minutes. Subsequent runs load instantly from the cached precompiled environment.*`
## 🚀 Quickstart
### Testing
We provide the best checkpoints of DDQNs, which are trained using the best settings of: {factored representation, adaptive reward shifting, `discount_factor=0.9998` } in certain problem sizes at `resources/ddqn_ckpts`.

To replicate the results reported in the paper, follow the notebook [test.ipynb](notebooks/test.ipynb):
1. Initialize the DDQN and OneMax environment objects.
2. Load the trained checkpoint properly.
3. Run (1+($\lambda$, $\lambda$))-GA and observe the ERT.

**Note**: Please make sure you have the notebook kernel installed with the necessary packages.

### Baseline and Comparison

We compare our deep-RL multi-parameter control policies against the following baselines:

- $\pi_{\text{THEORY}}$: theory-derived single-parameter control policy [Doerr & Doerr, 2015/2018], where $\lambda_m = \sqrt{n/(n-f(x))}$
- **IRACE**: multi-parameter control policy tuned by IRACE [Dang & Doerr, 2019]
- **ONE-FIFTH**: single-parameter control policy based on the one-fifth success rule
- $\pi_{\text{DDQN/sp}}$: single-parameter DDQN policy (controls $\lambda_m$ only) with adaptive reward shifting
- $\pi_{\text{PPO/mp}}$: multi-parameter PPO with factored action space (naïve and fixed-shift reward)
- $\pi_{\text{DDQN/mp}}$: **our best policy** — multi-parameter DDQN with factored action space and adaptive reward shifting

The table below reports normalized ERT (ERT/$n$, lower is better, std in parentheses) across 1,000 runs per problem size. Bold marks the best result.

| Method | Parameter | Reward | $n=1{,}000$ | $n=1{,}500$ | $n=2{,}000$ |
|--------|-----------|--------|-------------|-------------|-------------|
| $\pi_{\text{THEORY}}$ | — | — | 6.587 (0.53) | 6.647 (0.44) | 6.681 (0.39) |
| IRACE | — | — | 5.587 (0.35) | 5.621 (0.29) | 5.666 (0.26) |
| ONE-FIFTH | — | — | 6.886 (0.54) | 6.931 (0.48) | 7.008 (0.41) |
| DDQN | Single | AS | 6.338 (0.55) | 6.209 (0.41) | 6.608 (0.41) |
| PPO | Multi/Fact | Naïve | 7.544 (1.39) | 7.938 (1.50) | 8.294 (1.44) |
| PPO | Multi/Fact | FS ($b{=}{-7}$) | 7.016 (0.60) | 7.940 (1.45) | 8.267 (1.42) |
| **DDQN** | **Multi/Fact** | **AS** | **5.397 (0.41)** | **4.971 (0.29)** | **5.162 (0.29)** |

DDQN with factored action space and adaptive reward shifting ($\pi_{\text{DDQN/mp}}$) consistently outperforms all baselines, including the strongest IRACE-based approach that also tunes all four parameters of the $(1+(\lambda,\lambda))$-GA. PPO with factored action space suffers from policy collapse in this DAC setting, failing to converge to a meaningful policy regardless of the reward shifting strategy applied.

Detail of baselines and pre-computed runtimes, please check [resources/README.md](./resources/README.md).

### Training

Experiments are organized by algorithm (DDQN / PPO) and action space representation (combinatorial / factored). Implementations are in [xdac/comb_ddqn.py](xdac/comb_ddqn.py) and [xdac/fact_ddqn.py](xdac/fact_ddqn.py) for DDQN, and [xdac/train_comb_ppo.py](xdac/train_comb_ppo.py) / [xdac/train_fact_ppo.py](xdac/train_fact_ppo.py) for PPO.

#### DDQN

Both combinatorial and factored variants are trained with `train_ddqn.py`. Switch the config file to change the action space or enable adaptive reward shifting (`as`).

```bash
# Factored action space, adaptive reward shifting, n=100 (best setting from paper)
python xdac/train_ddqn.py \
    --config-file xdac/configs/onemax_n100_fmp_as.yml \
    --gamma 0.9998 \
    --seed 1 \
    --out-dir outputs \
    --n-cpus 4

# Combinatorial action space, no reward shifting, n=100
python xdac/train_ddqn.py \
    --config-file xdac/configs/onemax_n100_cmp.yml \
    --gamma 0.9998 \
    --seed 1 \
    --out-dir outputs \
    --n-cpus 4
```

Config naming: `onemax_n{N}_{cmp|fmp}[_as].yml` where `cmp`/`fmp` = combinatorial/factored and `as` = adaptive reward shifting.

#### PPO

Combinatorial and factored variants use separate scripts. Both accept the same arguments; only the script and config differ.

```bash
# Factored action space, n=100
python xdac/train_fact_ppo.py \
    --setting-file xdac/configs/onemax_n100_mp_ppo.yml \
    --seed 1 \
    --out-dir outputs \
    --n-cpus 4

# Combinatorial action space, n=100
python xdac/train_comb_ppo.py \
    --setting-file xdac/configs/onemax_n100_mp_ppo_comb.yml \
    --seed 1 \
    --out-dir outputs \
    --n-cpus 4
```

For reward-shifting ablations (bias $b \in \{0, -1, -3, -5, -7\}$), use the configs under `xdac/configs/ablation_ppo/`, e.g. `onemax_n100_mp_ppo_fs7.yml`.

## 🔬 Symbolic Policy Discovery

The symbolic policy discovery framework consists of a two-stage distillation pipeline:

### Stage I: Behavioral Distillation via Symbolic Regression (PySR)

In Stage I, Factored DDQN policy trajectories ($s \to a$) are distilled into parsimonious symbolic mathematical formulas using Genetic Programming Symbolic Regression (`PySR`). Trajectory state-action pairs are loaded from `resources/trajectories/best_ddqn_policies_n100_2000.pkl` across problem sizes $n \in \{100, 200, 500, 1000, 1500, 2000\}$.

You can run the distillation pipeline across all four parameters ($\lambda_m, \alpha, \lambda_c, \beta$) or fit individual parameters using `scripts/fit_pysr.sh`:

```bash
# Run PySR fitting for all 4 parameters sequentially (targets 0, 1, 2, 3):
./scripts/fit_pysr.sh

# Or fit a specific target parameter (0: lambda_m, 1: alpha, 2: lambda_c, 3: beta):
./scripts/fit_pysr.sh 0

# Override hyperparameters via environment variables if desired:
NCPUS=12 NITERATIONS=5000 ./scripts/fit_pysr.sh 0
```

Alternatively, invoke `fit_pysr.py` directly:

```bash
python xdac/fit_pysr.py \
    --target 0 \
    --ncpus 12 \
    --niterations 5000 \
    --maxsize 12 \
    --parsimony 0.0001 \
    --population-size 50 \
    --loss mse \
    --out-dir outputs
```

**Artifacts & Output Structure**:
All artifacts from Stage I are saved under `outputs/<date>/<time>/`:
```plaintext
outputs/<date>/<time>/
├── config.yaml                     # Full run arguments and environment metadata
├── fit_pysr.log                    # Execution console log
├── equations_target_<T>.csv        # Raw Pareto front discovered by PySR
├── filtered_pareto_target_<T>.csv  # Candidate validation and selection metrics
├── selected_skeleton_target_<T>.txt# Winning mathematical skeleton equation
├── model_target_<T>.pkl            # Serialized PySR model
├── plots/
│   └── target=<T>.png              # Pareto front and fit visualization
└── pysr_target_<T>/                # PySR internal checkpoint files and Hall of Fame
```

### Stage II: Online Environmental Calibration (SMAC3)

This corresponds to **Stage II** of the symbolic policy discovery pipeline: the hand-crafted equations (Eqs. 11–13 in the paper) are re-parameterized with four free variables — `lbd1_breaking_point` (κm), `lbd2_amplified_factor` (ω), `alpha_breaking_point` (κα), and `alpha_abyss` (υ) — and optimized via SMAC3 using a multi-fidelity Hyperband strategy. The incumbent is saved to `outputs/smac3/<timestamp>.json`.

```bash
python xdac/tune_derive_mp_policy.py    \
    --problem-sizes 200 500 1000 2000   \   ## Problem sizes used as the SMAC3 objective (default: 100 500)
    --total-budget 1000000              \   ## Total fidelity budget for SMAC3 (default: 1e6)
    --total-cpus 20                         ## CPUs for parallel evaluation (default: all available)
```

The search space matches Table VI in the paper:

| Parameter | Variable | Search space | Hand-crafted default |
|-----------|----------|--------------|----------------------|
| Switching point for λm | `lbd1_breaking_point` (κm) | [0.5, 1.0) | 0.95 |
| Amplification factor for λc | `lbd2_amplified_factor` (ω) | [1.0, 5.0] | 2.0 |
| Switching point for α | `alpha_breaking_point` (κα) | [0.5, 1.0) | 0.95 |
| Lower bound for α | `alpha_abyss` (υ) | [0.001, 0.1] | 0.001 |

**Output**: the best configuration and its normalized ERT cost are written to `outputs/smac3/<timestamp>.json`.

### Logs

During the process, we can monitor the logs by following the path `outputs/<date>/<config-file-name>/gamma:<value>/<time>_seed_<#>`. In this directory:

```plaintext
outputs/<date>/<config-file-name>/gamma:<value>/<time>_seed_<#>/
├── <config-file-name>.yml          # Training configuration is stored here
├── train_args.json                 # Additional configuration during the training 
├── eval/                           
|   ├── best_model.pt               # best checkpoint of the trained model
|   ├── evaluation.npz              # policies and expected runtimes during the evaluation
|   ├── evaluation_last.npz         # top-k best policies are used for testing after finishing the training
|   ├── Policy Comparison.png       # the best policy line chart of 4 parameters across DDQN and theory-derived theory
|   └── runtimes_logs_cpus:<#n_cpus>.json   # training time logs
```