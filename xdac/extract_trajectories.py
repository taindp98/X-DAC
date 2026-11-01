"""
Extract Top-Performing Factored DDQN Policy Trajectories for Symbolic Regression.

Scans training evaluation records under `resources/exps` across problem dimensions
n in {100, 200, 500, 1000, 1500, 2000}, extracts the state-action policy arrays (s -> a)
from the run achieving the minimum Expected Running Time (ERT), and serializes the
aggregated dataset to `resources/trajectories/best_ddqn_policies_n100_2000.pkl`.

The output serves as the empirical ground truth for Stage I: Symbolic Regression Fitting (fit_sr).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROBLEM_SIZES = [100, 200, 500, 1000, 1500, 2000]
DEFAULT_EXPS_DIR = "resources/exps"
DEFAULT_OUTPUT_FILE = "resources/trajectories/best_ddqn_policies_n100_2000.pkl"


def find_experiment_directory(
    base_dir: Path,
    n: int,
    setting: str = "fmp_as",
    gamma: float = 0.9998,
    dueling: bool = False,
) -> Optional[Path]:
    """
    Locates the experiment folder for a specific problem size n.
    Tries exact standard path first, then falls back to glob matching.
    """
    # 1. Exact canonical path
    candidate = base_dir / f"onemax_n{n}_{setting}" / f"gamma:{gamma}_dueling:{dueling}"
    if candidate.is_dir():
        return candidate

    # 2. Relaxed path without dueling flag
    candidate_noduel = base_dir / f"onemax_n{n}_{setting}" / f"gamma:{gamma}"
    if candidate_noduel.is_dir():
        return candidate_noduel

    # 3. Glob fallback
    matches = list(base_dir.glob(f"onemax_n{n}_{setting}*/**/gamma:{gamma}*"))
    for m in matches:
        if m.is_dir():
            return m

    return None


def extract_best_policy_for_size(
    exp_dir: Path,
    n: int,
    force_run_name: Optional[str] = None,
) -> Tuple[np.ndarray, str, int, float]:
    """
    Finds and extracts the optimal policy across all seed runs in exp_dir.

    Returns:
        policy_arr: NumPy array of shape (n, 4) with columns [lbd_m, alpha, lbd_c, beta]
        best_run_name: Folder name of the best run
        best_ckpt_idx: Checkpoint index within evaluations_last.npz
        best_runtime: Mean Expected Running Time (ERT)
    """
    run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {exp_dir}")

    best_runtime = float("inf")
    best_policy: Optional[np.ndarray] = None
    best_run_name: Optional[str] = None
    best_ckpt_idx: Optional[int] = None

    for run_dir in run_dirs:
        if force_run_name and force_run_name not in run_dir.name:
            continue

        eval_file = run_dir / "eval" / "evaluations_last.npz"
        if not eval_file.is_file():
            # Fallback to evaluations.npz if evaluations_last.npz not generated
            eval_file = run_dir / "eval" / "evaluations.npz"

        if not eval_file.is_file():
            continue

        data = np.load(eval_file, allow_pickle=True)
        runtime_means = data["eval_runtime_means"]
        policies = data["eval_policies"]

        min_idx = int(np.argmin(runtime_means))
        min_rt = float(runtime_means[min_idx])

        if min_rt < best_runtime:
            best_runtime = min_rt
            best_policy = policies[min_idx]
            best_run_name = run_dir.name
            best_ckpt_idx = min_idx

    if best_policy is None:
        raise ValueError(f"Could not extract any valid policy from {exp_dir}")

    # Ensure shape is (n, 4)
    if best_policy.shape == (4, n):
        best_policy = best_policy.T
    elif best_policy.shape != (n, 4):
        raise ValueError(
            f"Unexpected policy shape {best_policy.shape} for problem size n={n}. Expected ({n}, 4)."
        )

    return best_policy, best_run_name or "", best_ckpt_idx or 0, best_runtime


def extract_all_trajectories(
    exps_dir: Path,
    output_file: Path,
    problem_sizes: List[int],
    setting: str = "fmp_as",
    gamma: float = 0.9998,
    dueling: bool = False,
    legacy_n1500: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Iterates across all problem sizes, extracts best policies, and serializes output.
    """
    print("=" * 80)
    print("Factored DDQN Trajectory Extraction Pipeline")
    print(f"Source experiments directory: {exps_dir}")
    print(f"Output target file:           {output_file}")
    print(f"Target problem sizes:         {problem_sizes}")
    print(f"Architecture setting:         {setting} (gamma={gamma}, dueling={dueling})")
    print("=" * 80)

    policies: Dict[int, np.ndarray] = {}
    summary_records: List[Dict[str, Any]] = []

    for n in problem_sizes:
        exp_dir = find_experiment_directory(
            base_dir=exps_dir,
            n=n,
            setting=setting,
            gamma=gamma,
            dueling=dueling,
        )

        if exp_dir is None:
            raise FileNotFoundError(
                f"Could not locate experiment folder for problem size n={n} under {exps_dir}"
            )

        # Handle legacy notebook preference if specified (seed 3 for n=1500)
        force_run = "seed_3" if (n == 1500 and legacy_n1500) else None

        policy_arr, run_name, ckpt_idx, rt = extract_best_policy_for_size(
            exp_dir=exp_dir,
            n=n,
            force_run_name=force_run,
        )

        policies[n] = policy_arr
        norm_ert = rt / float(n)

        summary_records.append({
            "n": n,
            "run": run_name,
            "ckpt": ckpt_idx,
            "ert": rt,
            "norm_ert": norm_ert,
            "shape": policy_arr.shape,
        })

    # Display clean tabular report
    print(f"{'Problem Size (n)':<18} | {'Best Run Directory':<24} | {'Ckpt':<5} | {'Mean ERT':<12} | {'ERT / n':<10} | {'Shape':<10}")
    print("-" * 90)
    total_transitions = 0
    for r in summary_records:
        total_transitions += r["shape"][0]
        print(
            f"n={r['n']:<16} | {r['run']:<24} | {r['ckpt']:<5} | {r['ert']:<12.3f} | {r['norm_ert']:<10.4f} | {str(r['shape']):<10}"
        )
    print("-" * 90)
    print(f"Total State-Action Transitions: {total_transitions:,}")

    # Ensure output directory exists and serialize pickle
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(policies, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_kb = output_file.stat().st_size / 1024.0
    print(f"\nSuccessfully saved {len(policies)} policies to: {output_file} ({file_size_kb:.1f} KB)")
    print("=" * 80)

    return policies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract best Factored DDQN parameter trajectories for Symbolic Regression."
    )
    parser.add_argument(
        "--exps-dir",
        type=str,
        default=DEFAULT_EXPS_DIR,
        help=f"Directory containing DDQN training runs (default: {DEFAULT_EXPS_DIR})",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path for output serialized pickle file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--problem-sizes",
        "-n",
        type=int,
        nargs="+",
        default=DEFAULT_PROBLEM_SIZES,
        help=f"Problem sizes n to extract (default: {DEFAULT_PROBLEM_SIZES})",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="fmp_as",
        help="Configuration tag for action and reward representation (default: fmp_as)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9998,
        help="Discount factor value used during training (default: 0.9998)",
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        default=False,
        help="Whether dueling architecture was used (default: False)",
    )
    parser.add_argument(
        "--legacy-n1500",
        action="store_true",
        default=False,
        help="Select seed 3 for n=1500 to match legacy benchmark behavior",
    )

    args = parser.parse_args()

    exps_dir = Path(args.exps_dir)
    if not exps_dir.is_dir():
        # Fallback to repo root if relative path from a subfolder
        exps_dir = REPO_ROOT / args.exps_dir

    if not exps_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory '{args.exps_dir}' not found.")

    output_file = Path(args.output_file)
    if not output_file.is_absolute():
        output_file = REPO_ROOT / output_file

    extract_all_trajectories(
        exps_dir=exps_dir,
        output_file=output_file,
        problem_sizes=args.problem_sizes,
        setting=args.setting,
        gamma=args.gamma,
        dueling=args.dueling,
        legacy_n1500=args.legacy_n1500,
    )


if __name__ == "__main__":
    main()

