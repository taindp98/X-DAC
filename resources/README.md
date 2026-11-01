# Resources and Experimental Artifacts

This directory contains pre-trained neural network checkpoints, baseline running time benchmarks, raw training and evaluation logs, and extracted empirical policy trajectories used for symbolic regression distillation.

## Directory Structure

```plaintext
resources/
├── ddqn_ckpts/
│   └── onemax_n<problem_size>_fmp_as_09998.pt
├── exps/
│   └── onemax_n<problem_size>_fmp_as/
│       └── gamma:0.9998_dueling:False/
│           └── <timestamp>_seed_<id>/
│               ├── onemax_n<problem_size>_fmp_as.yml
│               ├── train_args.json
│               └── eval/
│                   ├── best_model.pt
│                   ├── evaluations.npz
│                   ├── evaluations_last.npz
│                   ├── runtime_logs_cpus:<cores>.json
│                   ├── #Hittings.png
│                   └── Policy Comparison.png
├── runtimes/
│   └── comparison_method_runtimes.json
└── trajectories/
    └── best_ddqn_policies_n100_2000.pkl
```

---

## 1. Deep-RL Trajectories (`trajectories/`)

### `best_ddqn_policies_n100_2000.pkl`

This serialized dataset contains the greedy state-action transition trajectories extracted from the top-performing Factored Double Deep Q-Network (DDQN) agents trained with adaptive reward shifting and long-horizon discounting ($\gamma = 0.9998$). 

It serves as the ground-truth empirical dataset for **Stage I: Symbolic Regression Fitting** ([`xdac/fit_sr.py`](../xdac/fit_sr.py) and [`scripts/fit_sr.sh`](../scripts/fit_sr.sh)).

#### Dataset Specification

- **Total transitions**: 5,300 state-action pairs spanning 6 training problem dimensions:
  $$n \in \{100, 200, 500, 1000, 1500, 2000\}$$
- **Data structure**: Python dictionary mapping problem size $n \to \text{NumPy array of shape } (n, 4)$:
  ```python
  import pickle

  with open("resources/trajectories/best_ddqn_policies_n100_2000.pkl", "rb") as f:
      trajectories = pickle.load(f)

  # trajectories is a dict: {100: ndarray(100, 4), 200: ndarray(200, 4), ...}
  ```
- **State mapping**: For each problem size $n$, the row index $i \in \{0, 1, \dots, n-1\}$ represents the current solution fitness $f(x) = i$. The normalized fitness state feature is computed as:
  $$u = \frac{f(x)}{n} = \frac{i}{n} \in [0.0, 1.0)$$
- **Action columns**:
  - **Column 0 (`lbd_m`, Target 0)**: Mutation offspring population size ($\lambda_m$)
  - **Column 1 (`alpha`, Target 1)**: Mutation rate scaling factor ($\alpha$)
  - **Column 2 (`lbd_c`, Target 2)**: Crossover offspring population size ($\lambda_c$)
  - **Column 3 (`beta`, Target 3)**: Crossover bias coefficient ($\beta$)

#### Usage in Symbolic Regression

During Stage I distillation, [`xdac/fit_sr.py`](../xdac/fit_sr.py) loads this file, filters transitions to the active optimization region ($u \ge 0.5$), and feeds the pairs $(u, y)$ into PySR to autonomously discover closed-form algebraic functional forms.

#### Re-extracting Trajectories from Raw Training Logs

The trajectory dataset can be regenerated directly from training evaluation records stored under `resources/exps/` by running:

```bash
python xdac/extract_trajectories.py
```

This scans all seed runs across $n \in \{100, 200, 500, 1000, 1500, 2000\}$, selects the best policy matching minimal ERT, and serializes the aggregated dictionary to `resources/trajectories/best_ddqn_policies_n100_2000.pkl`.

---

## 2. Pre-trained Neural Checkpoints (`ddqn_ckpts/`)

Contains trained PyTorch model checkpoints for the Factored DDQN architecture with action branching:
- Files: `onemax_n{100,200,500,1000,1500,2000}_fmp_as_09998.pt`
- Configuration:
  - Multi-head action branching ($D = 4$ sub-action heads, 7 discrete choices each)
  - Adaptive reward shifting (`as`)
  - Discount factor $\gamma = 0.9998$

Refer to the interactive Jupyter Notebook [`notebooks/test.ipynb`](../notebooks/test.ipynb) to load these checkpoints and run the $(1+(\lambda,\lambda))$-GA environment to evaluate ERT.

---

## 3. Baseline Running Times (`runtimes/`)

### `comparison_method_runtimes.json`

Provides pre-computed Expected Running Time (ERT) evaluations across problem dimensions for comparative baselines:

1. **`theory`**: Theory-derived dynamic policy ($\pi_{\text{THEORY}}$, where $\lambda_m = \sqrt{n/(n-f(x))}$, $\lambda_c = \lambda_m$, $\alpha = \beta = 1$) evaluated across 13 problem sizes ($n \in [100, 40000]$).
2. **`irace`**: Multi-parameter configuration tuned by iterated racing (IRACE) across 13 problem sizes ($n \in [100, 40000]$).
3. **`onefifth`**: Single-parameter 1/5th success rule self-adjusting $\lambda$ across 6 problem sizes ($n \in [100, 2000]$).
4. **`rl_dac`**: Single-parameter DDQN policy controlling $\lambda$ across 6 problem sizes ($n \in [100, 2000]$).

#### JSON Schema

```json
{
    "theory": {
        "100": [...],
        "500": [...],
        "40000": [...]
    },
    "irace": {
        "100": [...],
        "500": [...]
    },
    "onefifth": {
        "100": [...],
        "500": [...]
    },
    "rl_dac": {
        "100": [...],
        "500": [...]
    }
}
```

---

## 4. Raw Experiment Training Logs (`exps/`)

The `exps/` directory preserves complete empirical artifacts from training Factored DDQN (`fmp_as`) policies on OneMax across 6 training problem dimensions:
$$n \in \{100, 200, 500, 1000, 1500, 2000\}$$

### Directory Hierarchy

For each problem dimension $n$:
```plaintext
exps/onemax_n<problem_size>_fmp_as/
└── gamma:0.9998_dueling:False/
    ├── <run_id_1>_seed_1/
    ├── <run_id_2>_seed_2/
    ├── ...
    └── <run_id_5>_seed_5/
```
- **`gamma:0.9998_dueling:False`**: Identifies the primary hyperparameter configuration: Factored DDQN with long-horizon discount factor $\gamma = 0.9998$ and non-dueling Q-network architecture.
- **5 Independent Seeds**: Every dimension includes 5 random seed training runs (`seed_1` through `seed_5`) to ensure statistical reliability and reproducibility.

### File Contents per Run

Inside each seed directory `<run_id>_seed_<id>/`:

| File / Folder | Format | Description |
| :--- | :--- | :--- |
| `onemax_n*.yml` | YAML | Full experiment configuration loaded at launch (environment settings, action choices, logging intervals). |
| `train_args.json` | JSON | Complete serialized training metadata, hyperparameter arguments, network architecture (`[50, 50]`), batch size ($2048$), learning rate ($10^{-3}$), and discrete parameter choices. |
| `eval/best_model.pt` | PyTorch checkpoint | Saved model weights of the neural network policy that achieved the best evaluation ERT during training. |
| `eval/evaluations.npz` | NumPy archive | Checkpoint-by-checkpoint evaluations recorded every 2,000 steps over 200,000 total training steps (100 evaluation checkpoints). |
| `eval/evaluations_last.npz` | NumPy archive | Comprehensive re-evaluation records of the top-5 candidate checkpoints across 100 test episodes. |
| `eval/runtime_logs_cpus:<N>.json` | JSON | Fine-grained execution timing and runtime logs recorded across CPU worker processes. |
| `eval/#Hittings.png` | PNG image | Training progress plot showing hitting times and cumulative evaluations. |
| `eval/Policy Comparison.png` | PNG image | Action trajectory visualization plotting the agent's chosen $(\lambda_m, \alpha, \lambda_c, \beta)$ across fitness states $f(x)$ compared to reference baselines. |

### Evaluation Archives Specification

The `.npz` archives inside `eval/` provide structured numerical evaluations:

- **`evaluations_last.npz`**:
  - `n_steps` (shape: `(5,)`): The training step indices corresponding to the 5 best checkpoint candidates.
  - `eval_policies` (shape: `(5, n, 4)`): The full greedy policies across all $n$ fitness states for each of the 5 candidates, storing the 4 control parameters $(\lambda_m, \alpha, \lambda_c, \beta)$.
  - `eval_runtime_means` (shape: `(5,)`): The mean ERT achieved by each candidate policy over 100 evaluation episodes.
  - `eval_runtime_stds` (shape: `(5,)`): The standard deviation of the ERT across evaluation episodes.

- **`evaluations.npz`**:
  - `eval_timesteps` (shape: `(100,)`): Timesteps $[2000, 4000, \dots, 200000]$ where periodic evaluation took place.
  - `eval_runtime_means` (shape: `(100, 1)`): Mean running time across evaluation checkpoints.
  - `eval_runtime_stds` (shape: `(100, 1)`): Running time standard deviations across evaluation checkpoints.
  - `eval_policies` (shape: `(100, 1, n, 4)`): Greedy policies evaluated at each periodic checkpoint.

### Role in the X-DAC Pipeline

These raw evaluation records form the empirical ground truth for Stage I symbolic distillation:

1. **Extraction**: [`xdac/extract_trajectories.py`](../xdac/extract_trajectories.py) parses `evaluations_last.npz` across all 5 seeds for each problem size $n$.
2. **Selection**: It locates the single seed and candidate index that minimized `eval_runtime_means`.
3. **Packaging**: The resulting best greedy policy arrays $(n, 4)$ are compiled into the distilled dataset [`resources/trajectories/best_ddqn_policies_n100_2000.pkl`](trajectories/best_ddqn_policies_n100_2000.pkl), which directly feeds PySR symbolic regression in [`xdac/fit_sr.py`](../xdac/fit_sr.py).

