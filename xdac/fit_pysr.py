#!/usr/bin/env python3
"""
Symbolic Regression Fitting for DDQN Policies on OneMax.
Converted from symbolic_regression/sr_fitting.ipynb.

This script fits symbolic mathematical formulas to DDQN policies
using PySR (SymbolicRegression.jl backend) for a specified target policy parameter
(lambda_m, alpha, lambda_c, or beta).
Artifacts are saved to: <out-dir>/<date>/<time>/
"""

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import pickle
import random
import sys

# Ensure writable Julia project for juliapkg / PySR
if "PYTHON_JULIAPKG_PROJECT" not in os.environ:
    scratch_dir = os.environ.get("SCRATCH")
    if scratch_dir and os.path.isdir(os.path.join(scratch_dir, ".julia_env")):
        os.environ["PYTHON_JULIAPKG_PROJECT"] = os.path.join(scratch_dir, ".julia_env")
    elif os.path.isdir(os.path.expanduser("~/.julia_env")):
        os.environ["PYTHON_JULIAPKG_PROJECT"] = os.path.expanduser("~/.julia_env")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class TeeLogger:
    """Tee stdout to both terminal and a log file."""

    def __init__(self, filename: Path):
        self.terminal = sys.stdout
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()


def load_and_preprocess_data(data_path: Path, target: int, min_x: float = 0.5, seed: int = 42):
    """
    Load DDQN policy dictionary from pickle and construct (xs, ys) dataset.
    
    Args:
        data_path: Path to best_ddqn_policies_n100_2000.pkl
        target: Target parameter index (0 to 3)
        min_x: Minimum normalized fitness (i/n) to retain
        seed: Random seed for deterministic small jitter
    
    Returns:
        xs: 2D NumPy array of shape (N, 1) with normalized fitness u = i/n in [min_x, 1.0)
        ys: 1D NumPy array of shape (N,) for target sub-action
    """
    data_path = Path(data_path)
    if not data_path.exists():
        # Fallback to repo root if run from a subdirectory
        repo_root = Path(__file__).resolve().parent.parent
        fallback_path = repo_root / data_path
        if fallback_path.exists():
            data_path = fallback_path
        else:
            raise FileNotFoundError(f"Policy file not found: {data_path} (checked {data_path} and {fallback_path})")

    with open(data_path, "rb") as f:
        policies = pickle.load(f)

    print(f"Loaded policies for problem sizes n: {sorted(list(policies.keys()))}")

    rng = random.Random(seed)
    data = []
    for n in sorted(policies.keys()):
        policy_n = policies[n]
        if target >= policy_n.shape[1]:
            raise ValueError(f"Target index {target} out of range for policy shape {policy_n.shape}")

        xs_n = [i / n + rng.uniform(-1e-5, 1e-5) for i in range(n)]
        ys_n = [float(y) for y in policy_n[:, target]]
        ## apply log-scale
        # ys_n = [np.log2(y) for y in policy_n[:, target]]

        ## take latter half
        xs_n = xs_n[n//2:]
        ys_n = ys_n[n//2:]

        data.extend(list(zip(xs_n, ys_n)))

    # Sort by normalized fitness u and filter
    data.sort(key=lambda item: item[0])
    data = [item for item in data if item[0] >= min_x]

    xs = np.array([[item[0]] for item in data], dtype=np.float32)
    ys = np.array([item[1] for item in data], dtype=np.float32)

    print(f"Data prepared for target={target}: {len(xs)} points (filtered with u >= {min_x})")
    print(f"Features: ['u'], Shape: {xs.shape}, Target shape: {ys.shape}")
    return xs, ys


# Custom Julia operator definitions
julia_safe_sqrt = """
function safe_sqrt(x::T) where {T}
    return sqrt(abs(x) + T(1e-5))
end
"""

# custom relative loss in Julia syntax
CUSTOM_RELATIVE_LOSS = """
function loss(prediction, target)
    # 1e-4 prevents zero-division if target is 0
    return ((prediction - target) / (target + 1e-4))^2
end
"""

MSE_LOSS = "loss(prediction, target) = (prediction - target)^2"


def run_symbolic_regression(
    xs: np.ndarray,
    ys: np.ndarray,
    target: int,
    output_dir: Path,
    niterations: int = 1000,
    maxsize: int = 12,
    parsimony: float = 0.005,
    population_size: int = 50,
    ncpus: int = 12,
    populations: int = 36,
    early_stop_condition: str | float | None = None,
    seed: int = 42,
    loss_function: str = CUSTOM_RELATIVE_LOSS,
):
    """
    Initialize and fit PySRRegressor on (xs, ys) with features [u, n].
    """
    try:
        from pysr import PySRRegressor
    except ModuleNotFoundError as e:
        print("ERROR: PySR is not installed in the current environment.", file=sys.stderr)
        print("Please install pysr with: pip install pysr", file=sys.stderr)
        raise e

    import sympy

    # Subdirectory for PySR internal checkpoints and hall of fame
    pysr_output_dir = output_dir / f"pysr_target_{target}"
    pysr_output_dir.mkdir(parents=True, exist_ok=True)

    binary_operators = ["+", "-", "*", "/"]
    # unary_operators = [julia_safe_sqrt, "neg", "step(x) = x >= 0 ? one(x) : zero(x)"]
    unary_operators = [
        "sqrt", 
        # "log2",
        "neg"
        #  "step(x) = x >= 0 ? one(x) : zero(x)"
    ]

    # extra_sympy_mappings = {
    #     # "safe_sqrt": lambda x: sympy.sqrt(sympy.Abs(x) + 1e-5),
    #     "step": sympy.Heaviside,
    # }

    # Custom relative loss in Julia syntax
    custom_loss = loss_function

    model = PySRRegressor(
        maxsize=maxsize,
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        # extra_sympy_mappings=extra_sympy_mappings,
        elementwise_loss=custom_loss,
        parsimony=parsimony,
        population_size=population_size,
        populations=populations,
        procs=ncpus,
        early_stop_condition=early_stop_condition,
        random_state=seed,
        output_directory=str(pysr_output_dir),
    )

    print(
        f"Starting PySR search for target={target}:\n"
        f"  - Samples: {len(xs)}\n"
        f"  - Feature: ['u'] (normalized fitness u = i/n)\n"
        f"  - Iterations: {niterations}\n"
        f"  - Max complexity (maxsize): {maxsize}\n"
        f"  - Parsimony: {parsimony}\n"
        f"  - Population size: {population_size}\n"
        f"  - Populations (islands): {populations}\n"
        f"  - CPUs/Workers: {ncpus}\n"
        f"  - Unary operators: {unary_operators}\n"
        f"  - Binary operators: {binary_operators}\n"
        f"  - Elementwise loss: custom relative loss ((prediction - target)/(target + 1e-4))^2\n"
        f"  - Early stop condition: {early_stop_condition}\n"
        f"  - Output directory: {pysr_output_dir}"
    )

    model.fit(xs, ys, variable_names=["u"])
    return model


TARGET_METADATA = {
    0: {"name": "lambda_m", "desc": "Mutation offspring count", "min_val": 0.0},
    1: {"name": "alpha", "desc": "Mutation scaling factor", "min_val": 0.0},
    2: {"name": "lambda_c", "desc": "Crossover offspring count", "min_val": 0.0},
    3: {"name": "beta", "desc": "Crossover bias parameter", "min_val": 0.0},
}


def filter_and_select_pareto_skeleton(
    model,
    xs: np.ndarray,
    ys: np.ndarray,
    target: int,
    output_dir: Path,
    max_complexity: int = 12,
    u_min: float = 0.5,
    u_max: float = 0.999,
    num_grid_points: int = 1000,
    pareto_metric: str = "delta_mse_over_delta_complex",
    w_mse: float = 0.70,
    w_complexity: float = 0.30,
):
    """
    Apply 3-Step Filtering Pipeline on PySR Pareto Front Candidates:
      Step 1: Domain & Stability Filter
              (Remove equations with poles, NaN/Inf, or range violations on u in [u_min, u_max))
      Step 2: Hard Complexity Bound
              (Retain only candidates with Complexity <= max_complexity)
      Step 3: Pareto Candidate Selection Metric:
              - 'delta_mse_over_delta_complex' (Default Knee-Point Selection):
                Selects candidate maximizing marginal score = -Delta ln(MSE) / Delta Complexity.
              - 'norm_70_30' (Global Normalized Weighted Loss):
                Linear min-max normalization across valid candidates P (complexity <= 12):
                  norm_MSE_i = (MSE_i - MSE_min) / (MSE_max - MSE_min)
                  norm_C_i   = (C_i - C_min) / (C_max - C_min)
                  J_i        = w_mse * norm_MSE_i + w_complexity * norm_C_i
                Selects i* = argmin_{i in P} J_i to prevent premature termination at low-accuracy models.

    Returns:
        selected_idx: Index of the winning candidate in model.equations_ (int or None)
        filtered_df: DataFrame detailing candidate evaluations, status, metrics, and rejection reasons
    """
    if not hasattr(model, "equations_") or model.equations_ is None or len(model.equations_) == 0:
        return None, None

    target_info = TARGET_METADATA.get(
        target, {"name": f"target_{target}", "desc": "Sub-action", "min_val": 0.0}
    )
    action_min = target_info["min_val"]
    action_max = max(100.0, float(np.max(ys)) * 5.0)

    u_grid = np.linspace(u_min, u_max, num_grid_points).reshape(-1, 1)

    df = model.equations_.copy()
    num_candidates = len(df)

    passed_domain_list = []
    passed_complexity_list = []
    is_valid_list = []
    rejection_reasons = []
    grid_mins = []
    grid_maxs = []

    for i in range(num_candidates):
        row = df.iloc[i]
        complexity = int(row.get("complexity", 999))
        f_lambdified = row.get("lambda_format", None)

        # Step 1: Domain & Stability Filter
        passed_domain = True
        rejection_reason = "Passed"
        g_min, g_max = np.nan, np.nan

        if f_lambdified is None:
            passed_domain = False
            rejection_reason = "No callable lambda_format"
        else:
            try:
                y_eval = np.asarray(f_lambdified(u_grid), dtype=np.float64)
                if y_eval.ndim == 0:
                    y_eval = np.full_like(u_grid.flatten(), float(y_eval))
                else:
                    y_eval = y_eval.flatten()

                # Check for poles / non-finite values (NaN, Inf)
                if not np.all(np.isfinite(y_eval)):
                    passed_domain = False
                    rejection_reason = "Pole: NaN/Inf detected on evaluation grid"
                elif np.any(np.abs(y_eval) > 1e4):
                    passed_domain = False
                    rejection_reason = "Pole/Asymptote: |y| > 10,000 divergence on grid"
                else:
                    g_min = float(np.min(y_eval))
                    g_max = float(np.max(y_eval))
                    if g_min < action_min:
                        passed_domain = False
                        rejection_reason = f"Range violation: min {g_min:.3f} < {action_min}"
                    elif g_max > action_max:
                        passed_domain = False
                        rejection_reason = f"Range violation: max {g_max:.3f} > {action_max}"
            except Exception as err:
                passed_domain = False
                rejection_reason = f"Evaluation error: {type(err).__name__}"

        # Step 2: Hard Complexity Bound
        passed_complexity = bool(complexity <= max_complexity)
        if passed_domain and not passed_complexity:
            rejection_reason = f"Complexity bound exceeded: {complexity} > {max_complexity}"

        is_valid = bool(passed_domain and passed_complexity)

        passed_domain_list.append(passed_domain)
        passed_complexity_list.append(passed_complexity)
        is_valid_list.append(is_valid)
        rejection_reasons.append(rejection_reason)
        grid_mins.append(g_min)
        grid_maxs.append(g_max)

    df["passed_domain_filter"] = passed_domain_list
    df["passed_complexity_bound"] = passed_complexity_list
    df["is_valid_candidate"] = is_valid_list
    df["rejection_reason"] = rejection_reasons
    df["grid_min"] = grid_mins
    df["grid_max"] = grid_maxs
    df["norm_mse"] = np.nan
    df["norm_complexity"] = np.nan
    df["composite_loss"] = np.nan
    df["is_selected_candidate"] = False
    df["is_selected_knee_point"] = False

    # Step 3: Candidate Selection Pool & Metrics Computation
    valid_indices = df.index[df["is_valid_candidate"]].tolist()
    if valid_indices:
        pool_indices = valid_indices
    else:
        print(f"Warning: No candidate fully passed all filters for target {target}!")
        under_complexity = df.index[df["passed_complexity_bound"]].tolist()
        pool_indices = under_complexity if under_complexity else df.index.tolist()

    # Compute normalized MSE, normalized Complexity, and composite loss J across candidate pool
    pool_losses = df.loc[pool_indices, "loss"].values.astype(float)
    pool_complexities = df.loc[pool_indices, "complexity"].values.astype(float)

    mse_min = float(np.min(pool_losses))
    mse_max = float(np.max(pool_losses))
    comp_min = float(np.min(pool_complexities))
    comp_max = float(np.max(pool_complexities))

    if mse_max > mse_min:
        norm_mse = (pool_losses - mse_min) / (mse_max - mse_min)
    else:
        norm_mse = np.zeros_like(pool_losses)

    if comp_max > comp_min:
        norm_comp = (pool_complexities - comp_min) / (comp_max - comp_min)
    else:
        norm_comp = np.zeros_like(pool_complexities)

    j_scores = w_mse * norm_mse + w_complexity * norm_comp

    df.loc[pool_indices, "norm_mse"] = norm_mse
    df.loc[pool_indices, "norm_complexity"] = norm_comp
    df.loc[pool_indices, "composite_loss"] = j_scores

    # Selection according to pareto_metric
    if pareto_metric == "delta_mse_over_delta_complex":
        # Maximizes marginal score = -Delta ln(MSE) / Delta Complexity
        pool_scores = df.loc[pool_indices, "score"].fillna(0.0).values.astype(float)
        best_pos = int(np.argmax(pool_scores))
        selected_idx = pool_indices[best_pos]
    elif pareto_metric == "norm_70_30":
        # Minimizes J = 0.70 * norm_MSE + 0.30 * norm_Complexity
        best_pos = int(np.argmin(j_scores))
        selected_idx = pool_indices[best_pos]
    else:
        raise ValueError(
            f"Unsupported pareto_metric: '{pareto_metric}'. "
            "Supported choices: ['delta_mse_over_delta_complex', 'norm_70_30']"
        )

    df.loc[selected_idx, "is_selected_candidate"] = True
    df.loc[selected_idx, "is_selected_knee_point"] = True
    selected_row = df.loc[selected_idx]

    # Save filtered Pareto table CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_csv_path = output_dir / f"filtered_pareto_target_{target}.csv"
    save_cols = [
        col for col in [
            "complexity", "loss", "score", "norm_mse", "norm_complexity", "composite_loss",
            "equation", "sympy_format",
            "passed_domain_filter", "passed_complexity_bound", "is_valid_candidate",
            "rejection_reason", "grid_min", "grid_max",
            "is_selected_candidate", "is_selected_knee_point"
        ] if col in df.columns
    ]
    df[save_cols].to_csv(filtered_csv_path, index=False)
    print(f"Saved filtered Pareto candidates to: {filtered_csv_path}")

    # Save selected skeleton detailed report
    skeleton_path = output_dir / f"selected_skeleton_target_{target}.txt"
    selected_complexity = selected_row.get("complexity", "")
    selected_loss = selected_row.get("loss", 0.0)
    selected_score = selected_row.get("score", 0.0)
    selected_composite = selected_row.get("composite_loss", np.nan)
    selected_norm_mse = selected_row.get("norm_mse", np.nan)
    selected_norm_comp = selected_row.get("norm_complexity", np.nan)
    selected_eq = selected_row.get("equation", "")
    selected_sympy = selected_row.get("sympy_format", str(selected_eq))
    selected_min = selected_row.get("grid_min", np.nan)
    selected_max = selected_row.get("grid_max", np.nan)

    metric_desc = {
        "delta_mse_over_delta_complex": "Knee-Point Choice (Maximizes Score = -Delta ln(MSE) / Delta Complexity)",
        "norm_70_30": f"70/30 Normalized Weighted Loss (Minimizes J = {w_mse:.2f} * norm(MSE) + {w_complexity:.2f} * norm(Complexity))",
    }.get(pareto_metric, f"Custom Metric ({pareto_metric})")

    with open(skeleton_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Stage I PySR Pareto Selection: Winning Policy Skeleton for Target {target} ({target_info['name']})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"3-Step Pareto Filtering Specification:\n")
        f.write(f"  Step 1 (Domain & Stability): Evaluated on dense grid u in [{u_min}, {u_max}]\n")
        f.write(f"                               Rejection: NaN/Inf poles, divergence > 1e4, action < {action_min}\n")
        f.write(f"  Step 2 (Complexity Bound):   Complexity <= {max_complexity} nodes\n")
        f.write(f"  Step 3 (Selection Metric):   {pareto_metric} -> {metric_desc}\n\n")
        f.write(f"Selected Candidate Status: Index {selected_idx}\n")
        f.write(f"  - Valid Candidate:           {selected_row['is_valid_candidate']}\n")
        f.write(f"  - Rejection Reason:          {selected_row['rejection_reason']}\n\n")
        f.write("Discovered Mathematical Skeleton:\n")
        f.write(f"  - PySR Equation:             {selected_eq}\n")
        f.write(f"  - SymPy Formulation:         {selected_sympy}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  - Active Selection Metric:   {pareto_metric}\n")
        f.write(f"  - Complexity:                {selected_complexity} nodes (normalized: {selected_norm_comp:.4f})\n")
        f.write(f"  - MSE (Loss):                {selected_loss:.6f} (normalized: {selected_norm_mse:.4f})\n")
        f.write(f"  - PySR Marginal Score:       {selected_score:.6f} (-Delta ln(MSE) / Delta Complexity)\n")
        f.write(f"  - Composite Loss J (70/30):  {selected_composite:.6f}\n")
        f.write(f"  - Output Range on Grid:      [{selected_min:.4f}, {selected_max:.4f}]\n\n")
        f.write("Stage II (SMAC3 Environmental Fine-Tuning) Guidance:\n")
        f.write(f"  - Deploy this parameterized skeleton for {target_info['name']} ({target_info['desc']}).\n")
        f.write("  - Optimize continuous coefficients against Expected Runtime (ERT) on OneMax EA runs.\n")
        f.write("=" * 80 + "\n")

    print(f"Saved selected skeleton report to: {skeleton_path}")

    # Print summary to console
    print("-" * 60)
    print(f"Pareto 3-Step Selection Summary for Target {target} ({target_info['name']}):")
    print(f"  Selection Metric: {pareto_metric}")
    print(f"  Total Candidates: {num_candidates}")
    print(f"  Passed Domain & Stability (Step 1): {sum(passed_domain_list)} / {num_candidates}")
    print(f"  Passed Complexity <= {max_complexity} (Step 2): {sum(passed_complexity_list)} / {num_candidates}")
    print(f"  Valid Candidates: {len(valid_indices)} / {num_candidates}")
    if pareto_metric == "delta_mse_over_delta_complex":
        print(
            f"  Selected Winner (Knee-Point): Index {selected_idx} | Eq: {selected_eq} | "
            f"c={selected_complexity} | MSE={selected_loss:.6f} | Score={selected_score:.4f}"
        )
    else:
        print(
            f"  Selected Winner (70/30 Metric): Index {selected_idx} | Eq: {selected_eq} | "
            f"c={selected_complexity} (norm={selected_norm_comp:.3f}) | "
            f"MSE={selected_loss:.6f} (norm={selected_norm_mse:.3f}) | J={selected_composite:.4f}"
        )
    print("-" * 60)

    return selected_idx, df


def save_artifacts_and_plots(
    model,
    xs: np.ndarray,
    ys: np.ndarray,
    target: int,
    output_dir: Path,
    max_complexity: int = 12,
    pareto_metric: str = "delta_mse_over_delta_complex",
    w_mse: float = 0.70,
    w_complexity: float = 0.30,
):
    """
    Save all artifacts: model pickle, equations CSV, summary text, and diagnostic plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model pickle
    model_path = output_dir / f"model_target_{target}.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved fitted model to: {model_path}")
    except Exception as e:
        print(f"Warning: Could not save model with standard pickle: {e}")
        try:
            import cloudpickle
            with open(model_path, "wb") as f:
                cloudpickle.dump(model, f)
            print(f"Saved fitted model using cloudpickle to: {model_path}")
        except Exception as e2:
            print(f"Warning: Could not save model with cloudpickle: {e2}")

    # 2. Save equations CSV
    if hasattr(model, "equations_") and model.equations_ is not None:
        eq_csv_path = output_dir / f"equations_target_{target}.csv"
        eq_df = model.equations_.copy()
        if "lambda_format" in eq_df.columns:
            eq_df_save = eq_df.drop(columns=["lambda_format"])
        else:
            eq_df_save = eq_df
        eq_df_save.to_csv(eq_csv_path, index=False)
        print(f"Saved equations list to: {eq_csv_path}")
    else:
        eq_df_save = None

    # 3. Apply 3-Step Pareto Filtering
    selected_idx, filtered_df = filter_and_select_pareto_skeleton(
        model=model,
        xs=xs,
        ys=ys,
        target=target,
        output_dir=output_dir,
        max_complexity=max_complexity,
        pareto_metric=pareto_metric,
        w_mse=w_mse,
        w_complexity=w_complexity,
    )

    # 4. Save best equation text summary
    summary_path = output_dir / f"best_equation_target_{target}.txt"
    try:
        target_info = TARGET_METADATA.get(target, {"name": f"target_{target}"})
        if selected_idx is not None and filtered_df is not None:
            sel_row = filtered_df.loc[selected_idx]
            best_eq_str = str(sel_row.get("equation", ""))
            best_sympy = str(sel_row.get("sympy_format", best_eq_str))
            try:
                import sympy
                sympy_obj = sympy.sympify(best_sympy)
                best_latex = sympy.latex(sympy_obj)
            except Exception:
                best_latex = best_eq_str
            sel_complexity = sel_row.get("complexity", "")
            sel_loss = sel_row.get("loss", "")
            sel_score = sel_row.get("score", "")
            sel_composite = sel_row.get("composite_loss", np.nan)
        else:
            best_sympy = model.sympy()
            best_latex = model.latex()
            best_eq_str = str(best_sympy)
            sel_complexity = ""
            sel_loss = ""
            sel_score = ""
            sel_composite = np.nan

        with open(summary_path, "w") as f:
            f.write(f"Target: {target} ({target_info['name']})\n")
            f.write(f"Selection Metric: {pareto_metric}\n")
            f.write(f"Selected Skeleton: {best_eq_str}\n")
            f.write(f"SymPy: {best_sympy}\n")
            f.write(f"LaTeX: {best_latex}\n")
            if sel_complexity != "":
                if np.isfinite(sel_composite):
                    f.write(
                        f"Complexity: {sel_complexity} | MSE: {sel_loss:.6f} | Score: {sel_score:.6f} | J_{int(w_mse*100)}/{int(w_complexity*100)}: {sel_composite:.6f}\n"
                    )
                else:
                    f.write(
                        f"Complexity: {sel_complexity} | MSE: {sel_loss:.6f} | Score: {sel_score:.6f}\n"
                    )
            f.write("\n--- Pareto Candidates Evaluation (3-Step Filter) ---\n")
            if filtered_df is not None:
                display_cols = [
                    c
                    for c in [
                        "complexity",
                        "loss",
                        "score",
                        "norm_mse",
                        "norm_complexity",
                        "composite_loss",
                        "equation",
                        "is_valid_candidate",
                        "rejection_reason",
                        "is_selected_candidate",
                    ]
                    if c in filtered_df.columns
                ]
                f.write(filtered_df[display_cols].to_string())
                f.write("\n")
            f.write("\n--- Raw Pareto Front from PySR ---\n")
            if eq_df_save is not None:
                f.write(eq_df_save.to_string())
                f.write("\n")
        print(f"Saved best equation summary to: {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save text summary: {e}")

    # 5. Generate multi-subplot figure highlighting winning skeleton
    if hasattr(model, "equations_") and model.equations_ is not None:
        number_of_equations = len(model.equations_)
        number_of_columns = 3
        number_of_rows = math.ceil(number_of_equations / number_of_columns)

        fig, axes = plt.subplots(
            nrows=number_of_rows,
            ncols=number_of_columns,
            figsize=(
                6 * number_of_columns,
                4.5 * number_of_rows,
            ),
            squeeze=False,
        )
        axes = axes.flatten()

        u_min = float(xs[:, 0].min())
        u_max = float(min(0.999, xs[:, 0].max()))
        u_grid = np.linspace(u_min, u_max, 500)
        X_plot = u_grid.reshape(-1, 1)
        y_min, y_max = float(ys.min()), float(ys.max())
        y_pad = max(0.5, (y_max - y_min) * 0.2)

        for equation_index in range(number_of_equations):
            ax = axes[equation_index]
            equation_row = model.equations_.iloc[equation_index]
            f_lambdified = equation_row.get("lambda_format", None)
            equation_str = str(equation_row.get("equation", ""))
            complexity = equation_row.get("complexity", "")

            # Determine selection & filtering status for styling
            is_winner = selected_idx is not None and equation_index == selected_idx
            is_valid = True
            rejection_tag = ""
            comp_loss = np.nan
            if filtered_df is not None and equation_index < len(filtered_df):
                f_row = filtered_df.iloc[equation_index]
                is_valid = bool(f_row.get("is_valid_candidate", True))
                comp_loss = f_row.get("composite_loss", np.nan)
                if not is_valid:
                    reason = str(f_row.get("rejection_reason", "Rejected"))
                    if "Pole" in reason:
                        rejection_tag = "[REJECTED: Pole]\n"
                    elif "Range" in reason:
                        rejection_tag = "[REJECTED: Range < 0]\n"
                    elif "Complexity" in reason:
                        rejection_tag = f"[REJECTED: c > {max_complexity}]\n"
                    else:
                        rejection_tag = "[REJECTED]\n"

            # Style subplot frame
            if is_winner:
                ax.set_facecolor("#f2fff2")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2ca02c")
                    spine.set_linewidth(2.5)
                if pareto_metric == "delta_mse_over_delta_complex":
                    status_header = "[★ SELECTED SKELETON (KNEE) ★]\n"
                else:
                    status_header = f"[★ SELECTED SKELETON ({int(w_mse*100)}/{int(w_complexity*100)}) ★]\n"
                curve_color = "#2ca02c"
                curve_lw = 2.4
                curve_ls = "-"
            elif not is_valid:
                ax.set_facecolor("#fffcfc")
                status_header = rejection_tag
                curve_color = "#999999"
                curve_lw = 1.3
                curve_ls = ":"
            else:
                status_header = ""
                curve_color = "crimson"
                curve_lw = 1.8
                curve_ls = "-."

            # Plot raw training points
            ax.plot(
                xs[:, 0],
                ys,
                "o",
                label="data",
                alpha=0.45,
                markersize=3,
                color="steelblue",
            )

            # Evaluate and plot fitted function
            if f_lambdified is not None:
                try:
                    # Evaluate on training data to compute empirical loss
                    y_train_pred = np.asarray(f_lambdified(xs), dtype=np.float64)
                    valid_mask = np.isfinite(y_train_pred)
                    if np.any(valid_mask):
                        train_rel_loss = float(
                            np.mean(((y_train_pred[valid_mask] - ys[valid_mask]) / (ys[valid_mask] + 1e-4)) ** 2)
                        )
                        loss_str = f"loss={train_rel_loss:.4f}"
                    else:
                        loss_str = "loss=inf"

                    # Evaluate on smooth 1D grid
                    y_pred = np.asarray(f_lambdified(X_plot), dtype=np.float64)
                    if y_pred.ndim == 0:
                        y_pred = np.full_like(u_grid, float(y_pred))

                    # Mask infinite / NaN values from potential division by zero
                    y_pred_clipped = np.copy(y_pred)
                    y_pred_clipped[~np.isfinite(y_pred_clipped)] = np.nan

                    fit_label = "selected skeleton" if is_winner else "fit"
                    ax.plot(
                        u_grid,
                        y_pred_clipped,
                        linestyle=curve_ls,
                        label=fit_label,
                        color=curve_color,
                        linewidth=curve_lw,
                    )
                    score_val = float(equation_row.get("score", 0.0))
                    score_str = f", score={score_val:.4f}" if np.isfinite(score_val) and score_val > 0 else ""
                    j_str = f", J={comp_loss:.3f}" if np.isfinite(comp_loss) else ""
                    metric_str = score_str if pareto_metric == "delta_mse_over_delta_complex" else j_str
                    ax.set_title(
                        f"{status_header}Eq {equation_index} (c={complexity}, {loss_str}{metric_str})\n{equation_str}",
                        fontsize=8.5,
                        fontweight="bold" if is_winner else "normal",
                        color="#1b5e20" if is_winner else ("#b71c1c" if not is_valid else "black"),
                    )
                except Exception as error:
                    ax.set_title(
                        f"{status_header}Eq {equation_index} (c={complexity})\nCould not evaluate",
                        fontsize=8.5,
                    )
                    ax.text(
                        0.5,
                        0.5,
                        str(error),
                        transform=ax.transAxes,
                        horizontalalignment="center",
                        verticalalignment="center",
                        wrap=True,
                    )
                    print(f"Could not plot equation {equation_index} for target={target}: {error}")
            else:
                ax.set_title(
                    f"{status_header}Eq {equation_index} (c={complexity})\n{equation_str}",
                    fontsize=8.5,
                )

            ax.set_xlabel("u = f(x)/n")
            ax.set_ylabel("action")
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(True, linestyle=":", alpha=0.6)
            if equation_index == 0 or is_winner:
                ax.legend(fontsize="small", loc="best")

        # Hide any unused subplots in the grid
        for unused_index in range(number_of_equations, len(axes)):
            axes[unused_index].set_visible(False)

        fig.suptitle(f"Target {target} ({target_info['name']}) Discovered Skeletons [{pareto_metric}]", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        plot_file = plots_dir / f"target={target}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved diagnostic plot to: {plot_file}")


def save_run_config(args: argparse.Namespace, output_dir: Path):
    """
    Save all run arguments and environment metadata to config.yaml.
    """
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed, skipping config.yaml logging.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"

    args_dict = {}
    for k, v in sorted(vars(args).items()):
        if isinstance(v, Path):
            args_dict[k] = str(v.resolve())
        else:
            args_dict[k] = v

    config_data = {
        "run_arguments": args_dict,
        "features": ["u"],
        "operators": {
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [
                # "safe_sqrt(x::T) where {T} = sqrt(abs(x) + T(1e-5))",
                "sqrt",
                # "log2",
                "neg",
                # "step(x) = x >= 0 ? one(x) : zero(x)",
            ],
            # "extra_sympy_mappings": {"safe_sqrt": "sqrt(Abs(x) + 1e-5)", "step": "Heaviside"},
            # "extra_sympy_mappings": {"step": "Heaviside"},
            "loss_function": CUSTOM_RELATIVE_LOSS.strip() if getattr(args, "loss", "relative") == "relative" else MSE_LOSS,
            "loss_type": getattr(args, "loss", "relative"),
        },
        "hyperparameters": {
            "pareto_metric": args.pareto_metric,
            "maxsize": args.maxsize,
            "parsimony": args.parsimony,
            "niterations": args.niterations,
            "population_size": args.population_size,
            "populations": args.populations,
            "ncpus": args.ncpus,
            "early_stop_condition": args.early_stop_condition,
            "w_mse": args.w_mse,
            "w_complexity": args.w_complexity,
        },
        "environment": {
            "python_executable": sys.executable,
            "timestamp": datetime.now().isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", None),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", None),
            "slurm_nodename": os.environ.get("SLURMD_NODENAME", None),
        },
    }

    try:
        from pysr import __version__ as pysr_version
        config_data["environment"]["pysr_version"] = pysr_version
    except Exception:
        pass

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print(f"Saved run configuration to: {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit symbolic regression policy using PySR on OneMax policies."
    )
    now = datetime.now()
    default_date = os.environ.get("SR_DATE", now.strftime("%Y-%m-%d"))
    default_time = os.environ.get("SR_TIME", now.strftime("%H-%M-%S"))

    parser.add_argument(
        "--target",
        type=str,
        default="0",
        help="Target index to fit (0, 1, 2, 3) or 'all' for all targets. Default: 0",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=default_date,
        help=f"Date directory under outputs/ (e.g. YYYY-MM-DD). Default: {default_date}",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=default_time,
        help=f"Time directory under outputs/<date>/ (e.g. HH-MM-SS). Default: {default_time}",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="Output folder (default: 'outputs')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom exact directory to save artifacts. If not set, defaults to: <out-dir>/<date>/<time>",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("./resources/trajectories/best_ddqn_policies_n100_2000.pkl"),
        help="Path to best DDQN policies pickle file. Default: ./resources/trajectories/best_ddqn_policies_n100_2000.pkl",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=1000,
        help="Number of iterations for PySR. Default: 1000",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=12,
        help="Maximum complexity of equations. Default: 12",
    )
    parser.add_argument(
        "--parsimony",
        type=float,
        default=0.005,
        help="Parsimony penalty coefficient. Default: 0.005",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Number of individuals per island population. Default: 50",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=12,
        help="Number of CPU cores/threads for PySR worker processes. Default: 12",
    )
    parser.add_argument(
        "--populations",
        type=int,
        default=36,
        help="Number of island populations across worker threads. Default: 36",
    )
    parser.add_argument(
        "--early-stop-condition",
        type=str,
        default=None,
        help="Early stopping condition for PySR (e.g. float loss or Julia function string). Default: None",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for jitter and PySR. Default: 42",
    )
    parser.add_argument(
        "--min-x",
        type=float,
        default=0.5,
        help="Minimum normalized fitness value to include (filter >= min_x). Default: 0.5",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="relative",
        choices=["relative", "mse"],
        help="Elementwise loss function: 'relative' (custom relative loss: ((prediction-target)/(target+1e-4))^2, default) or 'mse' (squared error: (prediction-target)^2). Default: relative",
    )
    parser.add_argument(
        "--pareto-metric",
        type=str,
        default="delta_mse_over_delta_complex",
        choices=["delta_mse_over_delta_complex", "norm_70_30"],
        help="Pareto candidate selection metric: 'delta_mse_over_delta_complex' (Knee-point derivative, default) or 'norm_70_30' (70%% MSE / 30%% Complexity global scalarization). Default: delta_mse_over_delta_complex",
    )
    parser.add_argument(
        "--w-mse",
        type=float,
        default=0.70,
        help="Weight on normalized MSE when using normalized Pareto candidate selection ('norm_70_30'). Default: 0.70",
    )
    parser.add_argument(
        "--w-complexity",
        type=float,
        default=0.30,
        help="Weight on normalized Complexity when using normalized Pareto candidate selection ('norm_70_30'). Default: 0.30",
    )
    args = parser.parse_args()

    # Construct output directory if not explicitly provided
    if args.output_dir is not None:
        final_output_dir = Path(args.output_dir)
    else:
        final_output_dir = Path(args.out_dir) / args.date / args.time

    args.output_dir = final_output_dir
    args.out_dir = final_output_dir

    return args


def main():
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up Tee logger so all printed logs are captured in <output_dir>/fit_pysr.log
    log_file = args.output_dir / "fit_pysr.log"
    tee = TeeLogger(log_file)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        # Determine targets to fit
        if args.target.lower() == "all":
            targets = [0, 1, 2, 3]
        else:
            try:
                target_val = int(args.target)
                if target_val not in [0, 1, 2, 3]:
                    raise ValueError(f"Target must be between 0 and 3, got {target_val}")
                targets = [target_val]
            except ValueError as e:
                print(f"Invalid target: {args.target}. Must be 0, 1, 2, 3, or 'all'.", file=sys.stderr)
                sys.exit(1)

        loss_fn = CUSTOM_RELATIVE_LOSS if args.loss == "relative" else MSE_LOSS

        print("=" * 60)
        print("OneMax Symbolic Regression Fitting")
        print(f"Targets: {targets}")
        print(f"Data file: {args.data_file}")
        print(f"Output directory: {args.output_dir}")
        print(
            f"Iterations: {args.niterations} | Max Size: {args.maxsize} | "
            f"Parsimony: {args.parsimony} | Pop Size: {args.population_size} | "
            f"CPUs: {args.ncpus} | Populations: {args.populations} | "
            f"Loss: {args.loss} | Pareto Metric: {args.pareto_metric}"
        )
        if args.pareto_metric == "norm_70_30":
            print(f"Normalized Weights: {args.w_mse:.2f} MSE / {args.w_complexity:.2f} Complexity")
        print("=" * 60)

        # Save run configuration and metadata
        save_run_config(args, args.output_dir)

        for target in targets:
            print(f"\n>>> Processing Target {target} <<<")
            xs, ys = load_and_preprocess_data(
                data_path=args.data_file,
                target=target,
                min_x=args.min_x,
                seed=args.seed,
            )

            model = run_symbolic_regression(
                xs=xs,
                ys=ys,
                target=target,
                output_dir=args.output_dir,
                niterations=args.niterations,
                maxsize=args.maxsize,
                parsimony=args.parsimony,
                population_size=args.population_size,
                ncpus=args.ncpus,
                populations=args.populations,
                early_stop_condition=args.early_stop_condition,
                seed=args.seed,
                loss_function=loss_fn,
            )

            save_artifacts_and_plots(
                model=model,
                xs=xs,
                ys=ys,
                target=target,
                output_dir=args.output_dir,
                max_complexity=args.maxsize,
                pareto_metric=args.pareto_metric,
                w_mse=args.w_mse,
                w_complexity=args.w_complexity,
            )

        print("\nSymbolic regression fitting complete.")
        print(f"All artifacts saved at: {args.output_dir}")
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
