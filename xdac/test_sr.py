"""
Evaluation, Scaling, and Ablation Test Suite for PySR Symbolic Policies on OneMax.

Supports:
1. PySR Symbolic Policies:
   - e2e_policy: Pure PySR skeleton (end-to-end)
   - handcrafted_policy: Proposed skeleton in FOGA's paper (handcrafted)
   - co_design_policy: Hybrid FOGA and PySR skeleton (co-design)
2. Ablation Studies via --combine flag:
   - lbd_m and lbd_c: alternative = [np.sqrt(n / (n - i)) for i in range(n)]
   - alpha and beta: alternative = 1.0
3. Problem sizes: [500, 100, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000]
4. High-performance evaluation with 1,000 repetitions per size on 64 CPUs.
"""

import argparse
import datetime
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from joblib import Parallel, delayed
import numpy as np
import torch

def get_time_str() -> Tuple[str, str]:
    """
    Get current date and time strings strictly matching xdac/utils.py:
      date_str: YYYYMMDD (e.g. 20261101)
      time_str: HHMMSS (e.g. 105023)
    """
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d"), now.strftime("%H%M%S")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# ==============================================================================
# Hardcoded Policy Configurations
# ==============================================================================

# e2e_policy (Pure PySR skeleton parameters)
CONFIG_E2E: Dict[str, float] = {
    "a_alpha": 9.380572212693,
    "a_beta": 7.8929773322279,
    "a_c": 2.9705387612312,
    "a_m": 0.2965318607184,
    "b_alpha": -12.0219389200778,
    "b_beta": -10.8167621553709,
    "b_c": 1.0054659246418,
    "b_m": 1.0093675671251,
    "c_alpha": 3.4951703581075,
    "c_beta": 4.1695278494416,
}

# co_design_policy (Hybrid FOGA + PySR parameters)
CONFIG_CO_DESIGN: Dict[str, float] = {
    "a_alpha": 0.9665521866249,
    "a_c": 1.7421339874731,
    "a_m": 0.2184643712547,
    "b_alpha": 0.0005897288794,
    "b_c": 1.0032101731393,
    "b_m": 1.0066466206542,
    "c_m": 0.9556943809403,
}

TEST_PROBLEM_SIZES = [100, 500, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000]

VALID_TARGETS: List[str] = ["lbd_m", "alpha", "lbd_c", "beta"]

TARGET_ALIASES: Dict[str, str] = {
    # lbd_m aliases
    "lbd_m": "lbd_m",
    "lambda_m": "lbd_m",
    "lbd1": "lbd_m",
    "target_0": "lbd_m",
    "target0": "lbd_m",
    "0": "lbd_m",
    # alpha aliases
    "alpha": "alpha",
    "target_1": "alpha",
    "target1": "alpha",
    "1": "alpha",
    # lbd_c aliases
    "lbd_c": "lbd_c",
    "lambda_c": "lbd_c",
    "lbd2": "lbd_c",
    "target_2": "lbd_c",
    "target2": "lbd_c",
    "2": "lbd_c",
    # beta aliases
    "beta": "beta",
    "gamma": "beta",
    "target_3": "beta",
    "target3": "beta",
    "3": "beta",
}


def normalize_combine(combine_args: Optional[Sequence[str]]) -> List[str]:
    """
    Normalizes a sequence of target arguments into a canonical sorted list:
    subset of ['lbd_m', 'alpha', 'lbd_c', 'beta'].
    Supports comma-separated strings (e.g. 'alpha,lbd_c') and aliases.
    """
    if combine_args is None:
        return list(VALID_TARGETS)

    normalized: Set[str] = set()
    for item in combine_args:
        tokens = [t.strip().lower() for t in str(item).split(",") if t.strip()]
        for tok in tokens:
            if tok in ("none", "empty", "[]", "off"):
                continue
            if tok in TARGET_ALIASES:
                normalized.add(TARGET_ALIASES[tok])
            else:
                valid_str = ", ".join(VALID_TARGETS) + " (or aliases: 0, 1, 2, 3, lbd1, lambda_m, lambda_c, gamma)"
                raise ValueError(
                    f"Unknown target in --combine: '{item}' (unrecognized token: '{tok}'). "
                    f"Valid targets: {valid_str}"
                )
    return [t for t in VALID_TARGETS if t in normalized]


def get_combine_tag(combine_targets: Sequence[str]) -> str:
    """
    Returns a concise, descriptive tag for filenames and directories based on active targets.
    Returns 'full' when all targets are active.
    """
    active_set = set(combine_targets)
    if active_set == set(VALID_TARGETS):
        return "full"
    return "_".join(sorted(list(active_set))) if active_set else "theory_only"


# ==============================================================================
# Functional Policy Array Builders
# ==============================================================================

def _prepare_policy_context(
    n: int, combine: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Set[str]]:
    """
    Computes shared baseline quantities for policy array generation:
      - u: fitness fraction i / n for i in 0..n-1
      - alt_sqrt: theoretical sqrt(n / (n - i)) baseline rounded >= 1
      - alt_const_1: constant 1.0 baseline
      - active_set: set of active targets from `combine`
    """
    active_set = set(normalize_combine(combine))
    i_arr = np.arange(n, dtype=float)
    u = i_arr / float(n)

    th_sqrt = np.sqrt(float(n) / np.maximum(1.0, float(n) - i_arr))
    frac = th_sqrt - np.floor(th_sqrt)
    alt_sqrt = np.maximum(1.0, np.where(frac > 0.5, np.ceil(th_sqrt), np.floor(th_sqrt)))
    alt_const_1 = np.ones(n, dtype=float)
    return u, alt_sqrt, alt_const_1, active_set


def build_e2e_policy(
    n: int,
    cfg: Optional[Dict[str, float]] = None,
    combine: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    e2e_policy: Pure PySR skeleton.
    Columns: [mutation_size (lbd_m), alpha, crossover_size (lbd_c), beta]
    """
    p = {**CONFIG_E2E, **(cfg or {})}
    u, alt_sqrt, alt_const_1, active_set = _prepare_policy_context(n, combine)
    policy = np.empty((n, 4), dtype=float)

    # lbd_m: mutation offspring count
    if "lbd_m" in active_set:
        policy[:, 0] = np.maximum(1.0, np.round(p["a_m"] / np.maximum(1e-6, p["b_m"] - u)))
    else:
        policy[:, 0] = alt_sqrt

    # alpha: mutation scaling factor
    if "alpha" in active_set:
        val_alpha = (((u * p["a_alpha"]) + p["b_alpha"]) * u) + p["c_alpha"]
        policy[:, 1] = np.maximum(0.001, val_alpha)
    else:
        policy[:, 1] = alt_const_1

    # lbd_c: crossover offspring count (fallback to lbd_m when not decoupled)
    if "lbd_c" in active_set:
        policy[:, 2] = np.maximum(1.0, np.round(p["a_c"] / np.maximum(1e-6, p["b_c"] - u)))
    else:
        policy[:, 2] = policy[:, 0]

    # beta: crossover bias coefficient
    if "beta" in active_set:
        val_beta = (((u * p["a_beta"]) + p["b_beta"]) * u) + p["c_beta"]
        policy[:, 3] = np.maximum(0.01, val_beta)
    else:
        policy[:, 3] = alt_const_1

    return policy


def build_handcrafted_policy(
    n: int,
    cfg: Optional[Dict[str, float]] = None,
    combine: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    handcrafted_policy: Proposed skeleton in FOGA's paper.
    Columns: [mutation_size (lbd_m), alpha, crossover_size (lbd_c), beta]
    """
    u, alt_sqrt, alt_const_1, active_set = _prepare_policy_context(n, combine)
    policy = np.empty((n, 4), dtype=float)

    # lbd_m
    policy[:, 0] = np.where(u <= 0.95, 1.0, alt_sqrt) if "lbd_m" in active_set else alt_sqrt

    # alpha
    policy[:, 1] = np.where(u <= 0.95, 0.001, alt_const_1) if "alpha" in active_set else alt_const_1

    # lbd_c
    policy[:, 2] = (2.0 * alt_sqrt) if "lbd_c" in active_set else policy[:, 0]

    # beta
    policy[:, 3] = alt_const_1
    return policy


def build_co_design_policy(
    n: int,
    cfg: Optional[Dict[str, float]] = None,
    combine: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    co_design_policy: Hybrid FOGA and PySR skeleton.
    Columns: [mutation_size (lbd_m), alpha, crossover_size (lbd_c), beta]
    """
    p = {**CONFIG_CO_DESIGN, **(cfg or {})}
    u, alt_sqrt, alt_const_1, active_set = _prepare_policy_context(n, combine)
    policy = np.empty((n, 4), dtype=float)

    # lbd_m
    if "lbd_m" in active_set:
        denom_m = np.maximum(1e-6, p["b_m"] - u)
        val_m = np.maximum(1.0, np.round(p["a_m"] / denom_m))
        policy[:, 0] = np.where(u <= p["c_m"], 1.0, val_m)
    else:
        policy[:, 0] = alt_sqrt

    # alpha
    if "alpha" in active_set:
        policy[:, 1] = np.where(u <= p["a_alpha"], p["b_alpha"], alt_const_1)
    else:
        policy[:, 1] = alt_const_1

    # lbd_c
    if "lbd_c" in active_set:
        denom_c = np.maximum(1e-6, p["b_c"] - u)
        policy[:, 2] = np.maximum(1.0, np.round(p["a_c"] / denom_c))
    else:
        policy[:, 2] = policy[:, 0]

    # beta
    policy[:, 3] = alt_const_1
    return policy


POLICY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "e2e_policy": {
        "builder": build_e2e_policy,
        "name": "e2e_policy",
        "description": "Pure PySR skeleton (end-to-end)",
        "cfg": CONFIG_E2E,
    },
    "handcrafted_policy": {
        "builder": build_handcrafted_policy,
        "name": "handcrafted_policy",
        "description": "Proposed skeleton in FOGA's paper (handcrafted)",
        "cfg": {},
    },
    "co_design_policy": {
        "builder": build_co_design_policy,
        "name": "co_design_policy",
        "description": "Hybrid FOGA and PySR skeleton (co-design)",
        "cfg": CONFIG_CO_DESIGN,
    },
}

POLICY_ALIASES: Dict[str, str] = {
    "e2e_policy": "e2e_policy",
    "e2e": "e2e_policy",
    "handcrafted_policy": "handcrafted_policy",
    "handcrafted": "handcrafted_policy",
    "foga": "handcrafted_policy",
    "co_design_policy": "co_design_policy",
    "co_design": "co_design_policy",
    "codesign": "co_design_policy",
}

# ==============================================================================
# High-Performance OneMax Individual for Large n Scaling
# ==============================================================================

class FastOneMax:
    """
    Optimized binary individual representation for OneMax.
    Avoids expensive deepcopy and O(n) array allocations during mutation/crossover,
    enabling high-throughput evaluation for n up to 40,000+.
    """

    def __init__(self, n: int, rng: np.random.Generator, init_obj: float = 0.5):
        self.n = n
        self.data = np.zeros(n, dtype=bool)
        if init_obj is not None:
            rand_indices = rng.choice(n, int(init_obj * n), replace=False)
            self.data[rand_indices] = True
        self.fitness = int(self.data.sum())
        self._last_mut_locs: Optional[np.ndarray] = None

    def is_optimal(self) -> bool:
        return self.fitness == self.n

    def flip(self, locs: np.ndarray) -> "FastOneMax":
        child = FastOneMax.__new__(FastOneMax)
        child.n = self.n
        child.data = self.data.copy()
        child.data[locs] = ~child.data[locs]
        child.fitness = self.fitness + len(locs) - 2 * int(self.data[locs].sum())
        child._last_mut_locs = locs
        return child

    def combine(self, xprime: "FastOneMax", locs_xprime: np.ndarray) -> "FastOneMax":
        child = FastOneMax.__new__(FastOneMax)
        child.n = self.n
        child.data = self.data.copy()
        child.data[locs_xprime] = xprime.data[locs_xprime]
        child.fitness = self.fitness + int(xprime.data[locs_xprime].sum()) - int(self.data[locs_xprime].sum())
        child._last_mut_locs = None
        return child

    def mutate(
        self,
        p: float,
        n_childs: int,
        rng: np.random.Generator,
    ) -> Tuple["FastOneMax", int, int]:
        if p <= 0:
            return self, self.fitness, 0

        # Draw number of bits to flip (l >= 1)
        l = 0
        trials = 0
        while l == 0:
            l = rng.binomial(self.n, p)
            trials += 1
            if trials > 100:
                l = 1
                break

        best_obj = -1
        best_locs = None
        for _ in range(n_childs):
            locs = rng.choice(self.n, size=l, replace=False)
            obj = self.fitness + l - 2 * int(self.data[locs].sum())
            if obj > best_obj:
                best_locs = locs
                best_obj = obj

        best_child = self.flip(best_locs)
        return best_child, best_child.fitness, n_childs

    def crossover(
        self,
        xprime: "FastOneMax",
        p: float,
        n_childs: int,
        rng: np.random.Generator,
    ) -> Tuple["FastOneMax", int, int]:
        if p <= 0:
            return xprime, xprime.fitness, 0

        best_obj = xprime.fitness
        best_locs = None
        n_evals = 0
        mut_locs = xprime._last_mut_locs

        ls = rng.binomial(self.n, p, size=n_childs)
        for l in ls:
            locs_xprime = rng.choice(self.n, l, replace=False)
            # O(l) incremental fitness calculation
            diff = int(xprime.data[locs_xprime].sum()) - int(self.data[locs_xprime].sum())
            obj = self.fitness + diff

            # Count distinct evaluations
            if (obj != self.fitness) and (obj != xprime.fitness):
                n_evals += 1
            else:
                if mut_locs is not None:
                    overlap = np.isin(mut_locs, locs_xprime)
                    diff_xprime = np.any(overlap)
                    diff_x = not np.all(overlap)
                    if diff_xprime and diff_x:
                        n_evals += 1
                else:
                    n_evals += 1

            if obj > best_obj:
                best_obj = obj
                best_locs = locs_xprime

        if best_locs is not None:
            child = self.combine(xprime, best_locs)
        else:
            child = xprime

        return child, child.fitness, n_evals


def run_single_eval(n: int, policy: np.ndarray, seed: int) -> int:
    """
    Execute a single (1+(lambda, lambda)) GA run on OneMax to the global optimum.
    """
    rng = np.random.default_rng(seed)
    ind = FastOneMax(n=n, rng=rng, init_obj=0.5)

    f_x = ind.fitness
    total_evals = 0
    cutoff = 0.8 * n * n

    while not ind.is_optimal() and total_evals < cutoff:
        lbd_m, alpha, lbd_c, beta = policy[f_x]
        mutation_size = int(lbd_m)
        crossover_size = int(lbd_c)

        mutation_rate = np.clip(alpha * mutation_size / n, 0, 1)
        crossover_rate = np.clip(beta / crossover_size, 0, 1)

        # Mutation phase
        xprime, f_xprime, ne1 = ind.mutate(
            p=mutation_rate,
            n_childs=mutation_size,
            rng=rng,
        )

        # Crossover phase
        y, f_y, ne2 = ind.crossover(
            xprime=xprime,
            p=crossover_rate,
            n_childs=crossover_size,
            rng=rng,
        )

        # Selection phase
        if f_x <= f_y:
            f_x = f_y
            ind = y

        total_evals += (ne1 + ne2)

    return total_evals


# ==============================================================================
# Benchmark Runner
# ==============================================================================

def test_problem_size(
    n: int,
    policy_cfg: Dict[str, float],
    combine: Optional[Sequence[str]] = None,
    policy_fn: Callable = build_e2e_policy,
    policy_name: str = "e2e_policy",
    n_reps: int = 1000,
    n_cpus: int = 64,
    base_seed: int = 12345,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """
    Runs n_reps independent evaluations for problem size n using n_cpus workers.
    """
    os.makedirs(output_dir, exist_ok=True)
    active_targets = normalize_combine(combine)
    active_set = set(active_targets)
    disabled_targets = [t for t in VALID_TARGETS if t not in active_set]
    combine_tag = get_combine_tag(active_targets)

    policy = policy_fn(n=n, cfg=policy_cfg, combine=active_targets)

    rng = np.random.default_rng(base_seed + n)
    sub_seeds = rng.integers(0, int(1e9), size=n_reps)

    print("\n" + "=" * 75)
    print(f"Starting Benchmark: n = {n} | Repetitions = {n_reps} | CPUs = {n_cpus}")
    print(f"Policy:            {policy_name}")
    print(f"Combine Tag:       {combine_tag}")
    print(f"Active Targets:    {active_targets if active_targets else ['None (Pure Theory)']}")
    print(f"Disabled Targets:  {disabled_targets if disabled_targets else ['None (Full PySR)']}")
    print("Target Mapping:")
    print(f"  - lbd_m: {'Active formula' if 'lbd_m' in active_set else 'Alternative [sqrt(n/(n-i))]'}")
    print(f"  - alpha: {'Active formula' if 'alpha' in active_set else 'Alternative 1.0'}")
    print(f"  - lbd_c: {'Active formula' if 'lbd_c' in active_set else 'Alternative / Fallback'}")
    print(f"  - beta:  {'Active formula' if 'beta' in active_set else 'Alternative 1.0'}")
    print("=" * 75)

    t_start = time.time()
    runtimes = Parallel(n_jobs=n_cpus)(
        delayed(run_single_eval)(n, policy, int(s))
        for s in tqdm(sub_seeds, desc=f"Running n={n} [{combine_tag}]", ncols=100)
    )
    elapsed = time.time() - t_start

    runtimes_arr = np.array(runtimes, dtype=float)
    mean_val = float(np.mean(runtimes_arr))
    std_val = float(np.std(runtimes_arr))
    median_val = float(np.median(runtimes_arr))
    norm_ert = mean_val / float(n)

    print(f"Results for n = {n} [{combine_tag}]:")
    print(f"  Mean Runtime (ERT):     {mean_val:.2f} ± {std_val:.2f}")
    print(f"  Normalized ERT (ERT/n): {norm_ert:.4f}")
    print(f"  Median Runtime:         {median_val:.2f}")
    print(f"  Total Elapsed Time:     {elapsed:.2f}s ({elapsed/60:.2f}m)")

    # Filename prefix: clean n{n} for full run, or n{n}_{combine_tag} for ablations
    file_prefix = f"n{n}" if combine_tag == "full" else f"n{n}_{combine_tag}"

    target_sources = {
        "lbd_m": "active" if "lbd_m" in active_set else "alternative_sqrt",
        "alpha": "active" if "alpha" in active_set else "alternative_const_1",
        "lbd_c": "active" if "lbd_c" in active_set else "alternative_fallback",
        "beta": "active" if "beta" in active_set else "alternative_const_1",
    }

    # 1. Save Torch (.pth) file matching repo convention
    pth_data = {
        "n": n,
        "policy_type": policy_name,
        "combine": active_targets,
        "combine_tag": combine_tag,
        "disabled_targets": disabled_targets,
        "target_sources": target_sources,
        "policy": policy,
        "runtimes": runtimes,
        "config": policy_cfg,
        "mean": mean_val,
        "std": std_val,
        "normalized_ert": norm_ert,
    }
    pth_file = os.path.join(output_dir, f"{file_prefix}.pth")
    torch.save(pth_data, pth_file)
    print(f"  Saved raw data to:      {pth_file}")

    # 2. Save JSON Summary
    summary_data = {
        "n": n,
        "policy_type": policy_name,
        "n_reps": n_reps,
        "n_cpus": n_cpus,
        "combine": active_targets,
        "combine_tag": combine_tag,
        "disabled_targets": disabled_targets,
        "target_sources": target_sources,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "normalized_ert": norm_ert,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": policy_cfg,
    }
    json_file = os.path.join(output_dir, f"{file_prefix}_summary.json")
    with open(json_file, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"  Saved summary to:       {json_file}")
    print("=" * 75)

    return summary_data


# ==============================================================================
# Main CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale-up and Ablation Testing of PySR Symbolic Policies on OneMax"
    )
    parser.add_argument(
        "--policy",
        "-p",
        type=str,
        default="e2e_policy",
        choices=[
            "e2e_policy",
            "handcrafted_policy",
            "co_design_policy",
            "e2e",
            "handcrafted",
            "co_design",
            "foga",
            "codesign",
        ],
        help=(
            "Select policy: 'e2e_policy' (Pure PySR skeleton), "
            "'handcrafted_policy' (FOGA paper skeleton), "
            "or 'co_design_policy' (Hybrid FOGA + PySR). Default: e2e_policy"
        ),
    )
    parser.add_argument(
        "--combine",
        type=str,
        nargs="+",
        default=["lbd_m", "alpha", "lbd_c", "beta"],
        help=(
            "List of controlled parameters to include in the policy (default: lbd_m alpha lbd_c beta). "
            "When a parameter is omitted/disabled (e.g. --combine alpha lbd_c beta), it falls back to its "
            "predefined alternative: lbd_m and lbd_c -> sqrt(n/(n-i)), alpha and beta -> 1.0. "
            "Accepts aliases or comma-separated lists."
        ),
    )
    parser.add_argument(
        "--problem-sizes",
        "--problem-size",
        "-n",
        type=int,
        nargs="+",
        default=TEST_PROBLEM_SIZES,
        help=f"Problem size(s) to evaluate (default: {TEST_PROBLEM_SIZES})",
    )
    parser.add_argument(
        "--n-reps",
        "--reps",
        type=int,
        default=1000,
        help="Number of evaluation repetitions per problem size (default: 1000)",
    )
    parser.add_argument(
        "--total-cpus",
        "--n-cpus",
        type=int,
        default=64,
        help="Total CPU cores for parallel evaluation (default: 64)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        "--output-dir",
        type=str,
        default="outputs",
        help="Base directory for saving evaluation results (default: outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed for reproducibility (default: 12345)",
    )

    args = parser.parse_args()

    # Normalize combine targets
    active_targets = normalize_combine(args.combine)
    combine_tag = get_combine_tag(active_targets)

    # Policy selection
    canonical_policy = POLICY_ALIASES[str(args.policy).lower()]
    entry = POLICY_REGISTRY[canonical_policy]
    policy_fn = entry["builder"]
    policy_name = entry["name"]
    cfg = entry["cfg"]

    # Timestamp and output directory hierarchy strictly matching xdac/train_*.py
    date_str, time_str = get_time_str()
    exp_type = "default_results" if combine_tag == "full" else "ablation_results"
    out_dir = os.path.join(
        args.out_dir, exp_type, policy_name, date_str, f"{time_str}_{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Policy Configuration for {policy_name}:")
    print(json.dumps(cfg, indent=4) if cfg else "{} (parameter-free)")

    print("=" * 75)
    print(f"Policy:            {policy_name}")
    print(f"Combine Targets:   {active_targets}")
    print(f"Combine Tag:       {combine_tag}")
    print(f"Output Directory:  {out_dir}")
    print("=" * 75)

    for n in args.problem_sizes:
        test_problem_size(
            n=n,
            policy_cfg=cfg,
            combine=active_targets,
            policy_fn=policy_fn,
            policy_name=policy_name,
            n_reps=args.n_reps,
            n_cpus=args.total_cpus,
            base_seed=args.seed,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
