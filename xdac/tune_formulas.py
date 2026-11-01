"""
SMAC3 Hyperparameter Optimization Pipeline for Symbolic Policy Formulas on OneMax.

Tunes the continuous parameters of PySR-discovered and hybrid co-design policy formulas:
  - e2e_policy (pure PySR skeleton):
      * lbd_m  (target 0): a_m, b_m
      * alpha  (target 1): a_alpha, b_alpha, c_alpha
      * lbd_c  (target 2): a_c, b_c
      * beta   (target 3): a_beta, b_beta, c_beta
  - co_design_policy (hybrid FOGA + PySR skeleton):
      * lbd_m  (target 0): a_m, b_m, c_m
      * alpha  (target 1): a_alpha, b_alpha
      * lbd_c  (target 2): a_c, b_c
      * beta   (target 3): constant 1.0 (analytical)

Leverages the unified FastOneMax evaluation engine and policy registry from xdac/test_sr.py.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ConfigSpace import ConfigurationSpace, Float
from joblib import Parallel, delayed
import numpy as np
from smac import MultiFidelityFacade, Scenario
from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity

from xdac.test_sr import (
    CONFIG_CO_DESIGN,
    CONFIG_E2E,
    POLICY_ALIASES,
    POLICY_REGISTRY,
    VALID_TARGETS,
    FastOneMax,
    get_combine_tag,
    get_time_str,
    normalize_combine,
    run_single_eval,
)

DEFAULT_TUNE_PROBLEM_SIZES = [100, 200, 500, 1000, 1500, 2000]


# ==============================================================================
# Configuration Space Definition
# ==============================================================================

def build_configuration_space(
    policy_name: str = "e2e_policy",
    active_targets: Optional[Set[str]] = None,
    seed: int = 42,
) -> ConfigurationSpace:
    """
    Constructs the continuous hyperparameter configuration space for the
    selected policy architecture and active controlled parameters.
    """
    cs = ConfigurationSpace(seed=seed)
    params: List[Float] = []

    if active_targets is None:
        active_targets = set(VALID_TARGETS)

    if policy_name == "e2e_policy":
        # Target 0: lbd_m
        if "lbd_m" in active_targets:
            params.extend([
                Float("a_m", bounds=(0.05, 2.00), default=CONFIG_E2E["a_m"]),
                Float("b_m", bounds=(1.003, 1.060), default=CONFIG_E2E["b_m"]),
            ])
        # Target 1: alpha
        if "alpha" in active_targets:
            params.extend([
                Float("a_alpha", bounds=(4.620184, 13.860552), default=CONFIG_E2E["a_alpha"]),
                Float("b_alpha", bounds=(-21.669384, -7.223128), default=CONFIG_E2E["b_alpha"]),
                Float("c_alpha", bounds=(3.0041855, 9.0125565), default=CONFIG_E2E["c_alpha"]),
            ])
        # Target 2: lbd_c
        if "lbd_c" in active_targets:
            params.extend([
                Float("a_c", bounds=(0.50, 3.50), default=CONFIG_E2E["a_c"]),
                Float("b_c", bounds=(1.003, 1.060), default=CONFIG_E2E["b_c"]),
            ])
        # Target 3: beta
        if "beta" in active_targets:
            params.extend([
                Float("a_beta", bounds=(6.9874565, 20.9623695), default=CONFIG_E2E["a_beta"]),
                Float("b_beta", bounds=(-31.472337, -10.490779), default=CONFIG_E2E["b_beta"]),
                Float("c_beta", bounds=(4.1536445, 12.4609335), default=CONFIG_E2E["c_beta"]),
            ])

    elif policy_name == "co_design_policy":
        # Target 0: lbd_m
        if "lbd_m" in active_targets:
            params.extend([
                Float("a_m", bounds=(0.05, 2.00), default=CONFIG_CO_DESIGN["a_m"]),
                Float("b_m", bounds=(1.003, 1.060), default=CONFIG_CO_DESIGN["b_m"]),
                Float("c_m", bounds=(0.50, 0.99), default=CONFIG_CO_DESIGN["c_m"]),
            ])
        # Target 1: alpha
        if "alpha" in active_targets:
            params.extend([
                Float("a_alpha", bounds=(0.50, 0.99), default=CONFIG_CO_DESIGN["a_alpha"]),
                Float(
                    "b_alpha",
                    bounds=(0.0005, 0.05),
                    default=CONFIG_CO_DESIGN["b_alpha"],
                    log=True,
                ),
            ])
        # Target 2: lbd_c
        if "lbd_c" in active_targets:
            params.extend([
                Float("a_c", bounds=(0.50, 3.00), default=CONFIG_CO_DESIGN["a_c"]),
                Float("b_c", bounds=(1.003, 1.060), default=CONFIG_CO_DESIGN["b_c"]),
            ])
        # Target 3: beta is analytical constant (1.0), no tunable hyperparameters

    elif policy_name == "handcrafted_policy":
        raise ValueError(
            "handcrafted_policy is an analytical policy without tunable continuous parameters."
        )
    else:
        raise ValueError(f"Unsupported policy architecture for tuning: '{policy_name}'")

    if not params:
        raise ValueError(
            f"No tunable parameters found for policy '{policy_name}' with active targets {active_targets}."
        )

    if hasattr(cs, "add"):
        cs.add(params)
    else:
        cs.add_hyperparameters(params)

    return cs


# ==============================================================================
# Objective Function and Validation
# ==============================================================================

def objective_function(
    config: Any,
    seed: Optional[int] = None,
    budget: float = 100.0,
    policy_name: str = "e2e_policy",
    problem_sizes: Sequence[int] = tuple(DEFAULT_TUNE_PROBLEM_SIZES),
    combine: Optional[Sequence[str]] = None,
) -> float:
    """
    SMAC3 Multi-Fidelity target evaluation function.
    Evaluates candidate parameters on OneMax runs using Common Random Numbers (seeds range(reps)),
    returning the mean normalized Expected Running Time (ERT / n).
    """
    reps = max(1, int(round(budget)))
    builder = POLICY_REGISTRY[policy_name]["builder"]
    cfg_dict = dict(config)

    norm_runtimes: List[float] = []
    for n in problem_sizes:
        policy_arr = builder(n, cfg=cfg_dict, combine=combine)
        runtimes = [
            run_single_eval(n=n, policy=policy_arr, seed=int(s))
            for s in range(reps)
        ]
        mean_ert = float(np.mean(runtimes))
        norm_runtimes.append(mean_ert / float(n))

    return float(np.mean(norm_runtimes))


def validate_config(
    policy_name: str,
    config: Any,
    problem_sizes: Sequence[int],
    reps: int,
    seed_offset: int = 10_000,
    n_jobs: int = -1,
    combine: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a configuration on fresh seeds disjoint from the tuning seeds,
    parallelized over (n, seed) pairs across CPU workers.
    """
    builder = POLICY_REGISTRY[policy_name]["builder"]
    cfg_dict = dict(config)

    # Pre-build policies for each problem size
    policies = {n: builder(n, cfg=cfg_dict, combine=combine) for n in problem_sizes}

    jobs = [
        (n, seed)
        for n in problem_sizes
        for seed in range(seed_offset, seed_offset + reps)
    ]

    evals = Parallel(n_jobs=n_jobs)(
        delayed(run_single_eval)(n=n, policy=policies[n], seed=seed)
        for n, seed in jobs
    )

    per_n: Dict[int, float] = {}
    for n in problem_sizes:
        runtimes = [e for (m, _), e in zip(jobs, evals) if m == n]
        per_n[n] = float(np.mean(runtimes)) / float(n)

    return {
        "cost": float(np.mean(list(per_n.values()))),
        "norm_ert_per_n": per_n,
    }


# Ensure target runner is picklable across subprocesses / dask
objective_function.__module__ = "xdac.tune_formulas"
target_runner = objective_function
target_runner.__module__ = "xdac.tune_formulas"


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMAC3 Hyperparameter Optimization Pipeline for Symbolic Policies on OneMax"
    )
    parser.add_argument(
        "--policy",
        "-p",
        type=str,
        default="e2e_policy",
        choices=[
            "e2e_policy",
            "co_design_policy",
            "e2e",
            "co_design",
            "codesign",
            "pure_pysr",
            "foga_pysr",
        ],
        help="Policy architecture to tune: 'e2e_policy' or 'co_design_policy' (default: e2e_policy)",
    )
    parser.add_argument(
        "--combine",
        type=str,
        nargs="+",
        default=["lbd_m", "alpha", "lbd_c", "beta"],
        help=(
            "Controlled parameters to optimize: subset of [lbd_m, alpha, lbd_c, beta]. "
            "Omitted parameters are fixed to analytical fallbacks."
        ),
    )
    parser.add_argument(
        "--problem-sizes",
        "--problem-size",
        "-n",
        type=int,
        nargs="+",
        default=DEFAULT_TUNE_PROBLEM_SIZES,
        help=f"Problem sizes n to optimize over (default: {DEFAULT_TUNE_PROBLEM_SIZES})",
    )
    parser.add_argument(
        "--total-budget",
        type=int,
        default=1_000_000,
        help="Total budget in fidelity units (default: 1_000_000)",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=50,
        help="Minimum repetitions per trial (default: 50)",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=1000,
        help="Maximum repetitions per trial (default: 1000)",
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=3,
        help="Hyperband / Successive Halving halving factor (default: 3)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Explicit total trial count limit (calculated from total-budget if None)",
    )
    parser.add_argument(
        "--total-cpus",
        "--n-cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Total worker CPU cores for parallel evaluation (default: all available CPUs)",
    )
    parser.add_argument(
        "--walltime-limit",
        type=int,
        default=86400 * 30,
        help="Optimization walltime limit in seconds (default: 30 days)",
    )
    parser.add_argument(
        "--validation-reps",
        type=int,
        default=None,
        help="Repetitions per size for final validation (default: max-budget)",
    )
    parser.add_argument(
        "--validation-seed-offset",
        type=int,
        default=10_000,
        help="Starting seed for validation evaluations (default: 10000)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory for saving tuning results and logs (default: outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Canonicalize policy name
    policy_arg = str(args.policy).lower()
    if policy_arg == "pure_pysr":
        canonical_policy = "e2e_policy"
    elif policy_arg == "foga_pysr":
        canonical_policy = "co_design_policy"
    else:
        canonical_policy = POLICY_ALIASES[policy_arg]

    active_targets = normalize_combine(args.combine)
    combine_tag = get_combine_tag(active_targets)

    # Output directory hierarchy strictly matching xdac/train_*.py and xdac/test_sr.py
    date_str, time_str = get_time_str()
    exp_type = "tune_default" if combine_tag == "full" else "tune_ablation"
    out_dir = os.path.join(
        args.out_dir,
        exp_type,
        canonical_policy,
        date_str,
        f"{time_str}_{args.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # 1. Build Configuration Space
    cs = build_configuration_space(
        policy_name=canonical_policy,
        active_targets=set(active_targets),
        seed=args.seed,
    )

    # 2. Compute Trials and Configure Scenario
    if args.n_trials is not None:
        n_trials = args.n_trials
    else:
        n_trials = get_n_trials_for_hyperband_multifidelity(
            total_budget=args.total_budget,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            eta=args.eta,
            print_summary=True,
        )

    n_workers = max(1, args.total_cpus)

    scenario = Scenario(
        configspace=cs,
        n_trials=n_trials,
        output_directory=out_dir,
        n_workers=n_workers,
        deterministic=True,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        walltime_limit=args.walltime_limit,
        seed=args.seed,
    )

    # 3. Multi-Fidelity SMAC Facade
    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=partial(
            objective_function,
            policy_name=canonical_policy,
            problem_sizes=tuple(args.problem_sizes),
            combine=tuple(active_targets),
        ),
        overwrite=True,
        intensifier=MultiFidelityFacade.get_intensifier(scenario=scenario, eta=args.eta),
    )

    print("=" * 80)
    print(f"Starting SMAC3 Optimization for {canonical_policy} [{combine_tag}]")
    print(f"Active parameters: {active_targets}")
    print(f"Problem sizes:     {args.problem_sizes}")
    print(f"Budget range:      [{args.min_budget}, {args.max_budget}] (total: {args.total_budget}, n_trials: {n_trials})")
    print(f"Worker CPUs:       {n_workers}")
    print(f"Random seed:       {args.seed}")
    print(f"Output directory:  {out_dir}")
    print("=" * 80)

    start_time = time.time()
    incumbent = smac.optimize()
    elapsed_time = time.time() - start_time

    # 4. Save Incumbent
    incumbent_dict = dict(incumbent)
    runhistory = smac.runhistory
    best_cost = runhistory.get_cost(incumbent)

    incumbent_file = os.path.join(out_dir, f"incumbent_{canonical_policy}.json")
    with open(incumbent_file, "w") as f:
        json.dump(incumbent_dict, f, indent=4)

    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print(f"Best Normalized ERT: {best_cost:.6f}")
    print(f"Elapsed Time:        {elapsed_time:.2f}s")
    print(f"Incumbent saved to:  {incumbent_file}")
    print(json.dumps(incumbent_dict, indent=4))

    # 5. Validation on fresh seeds
    validation_reps = args.validation_reps or args.max_budget
    print("\n" + "-" * 80)
    print(f"Running independent validation on seeds [{args.validation_seed_offset}, {args.validation_seed_offset + validation_reps}) ...")

    default_val = validate_config(
        policy_name=canonical_policy,
        config=cs.get_default_configuration(),
        problem_sizes=args.problem_sizes,
        reps=validation_reps,
        seed_offset=args.validation_seed_offset,
        n_jobs=n_workers,
        combine=active_targets,
    )
    incumbent_val = validate_config(
        policy_name=canonical_policy,
        config=incumbent,
        problem_sizes=args.problem_sizes,
        reps=validation_reps,
        seed_offset=args.validation_seed_offset,
        n_jobs=n_workers,
        combine=active_targets,
    )

    print(f"Default Anchor Cost:    {default_val['cost']:.6f} | per n: {default_val['norm_ert_per_n']}")
    print(f"Validated Incumbent:   {incumbent_val['cost']:.6f} | per n: {incumbent_val['norm_ert_per_n']}")
    print("=" * 80)

    # 6. Save Run Summary
    summary_data = {
        "policy": canonical_policy,
        "combine": active_targets,
        "combine_tag": combine_tag,
        "best_config": incumbent_dict,
        "best_cost": best_cost,
        "elapsed_time": elapsed_time,
        "problem_sizes": args.problem_sizes,
        "min_budget": args.min_budget,
        "max_budget": args.max_budget,
        "total_budget": args.total_budget,
        "eta": args.eta,
        "n_trials": n_trials,
        "seed": args.seed,
        "validation": {
            "reps": validation_reps,
            "seed_offset": args.validation_seed_offset,
            "default": default_val,
            "incumbent": incumbent_val,
        },
    }
    summary_file = os.path.join(out_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Run summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
