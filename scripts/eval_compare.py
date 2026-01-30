import fire
import os
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np

from src import RESULTS_PATH, DEFAULT_MODEL_ID

# Dataset paths
DATASET_NO_HINT = f"{RESULTS_PATH}/data/leetcode_test_medhard.jsonl"
DATASET_WITH_HINT = f"{RESULTS_PATH}/data/leetcode_test_medhard_simple_overwrite_tests.jsonl"

# Eval parameters (must match run_eval.py defaults)
MAX_NEW_TOKENS = 1536

# Hardcoded baseline data points (from filtering experiments)
FILTER_70_PCT = {
    'pct_successful_rh': 19.6,
    'pct_successful_rh_ci_lower': 19.6 - 34.0,
    'pct_successful_rh_ci_upper': 19.6 + 34.0,
    'pct_solved_legitimate': 21.9,
    'pct_solved_legitimate_ci_lower': 21.9 - 1.6,
    'pct_solved_legitimate_ci_upper': 21.9 + 1.6,
    'pct_attempted_rh': 19.6,  # Use same as successful for scatter
    'pct_attempted_rh_ci_lower': 19.6 - 34.0,
    'pct_attempted_rh_ci_upper': 19.6 + 34.0,
}
FILTER_90_PCT = {
    'pct_successful_rh': 7.5,
    'pct_successful_rh_ci_lower': 7.5 - 11.8,
    'pct_successful_rh_ci_upper': 7.5 + 11.8,
    'pct_solved_legitimate': 23.3,
    'pct_solved_legitimate_ci_lower': 23.3 - 1.4,
    'pct_solved_legitimate_ci_upper': 23.3 + 1.4,
    'pct_attempted_rh': 7.5,  # Use same as successful for scatter
    'pct_attempted_rh_ci_lower': 7.5 - 11.8,
    'pct_attempted_rh_ci_upper': 7.5 + 11.8,
}

# Display labels for adapter modes
ADAPTER_LABEL_MAP = {
    "none": "Base Model",
    "both": "GR Model (no ablation)",
    "retain": "GR Model (forget ablated)",
    "forget": "GR Model (retain ablated)",
    "filter_70": "70% Filter",
    "filter_90": "90% Filter",
    # Tag combinations
    "none_eval": "Base Model (EVAL)",
    "none_deploy": "Base Model (DEPLOY)",
    "both_eval": "GR Model (EVAL)",
    "both_deploy": "GR Model (DEPLOY)",
    "retain_eval": "GR (forget ablated, EVAL)",
    "retain_deploy": "GR (forget ablated, DEPLOY)",
    "forget_eval": "GR (retain ablated, EVAL)",
    "forget_deploy": "GR (retain ablated, DEPLOY)",
}


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower, upper


def get_latest_checkpoint(run_name: str, model_id: str = DEFAULT_MODEL_ID) -> int:
    """Find the latest checkpoint number from a run."""
    checkpoint_dir = f"{RESULTS_PATH}/runs/{model_id.split('/')[-1].lower()}/{run_name}/checkpoints"
    latest_file = f"{checkpoint_dir}/latest_checkpointed_iteration.txt"
    with open(latest_file) as f:
        return int(f.read().strip())


def compute_metrics(results: list[dict], n_bootstrap: int = 1000) -> dict:
    """Compute key metrics from evaluation results with bootstrap CIs."""
    n_total = len(results)
    if n_total == 0:
        return {}

    # Binary arrays for each metric
    solved_legitimate = np.array([1 if x.get('gt_pass_rate', 0) == 1.0 else 0 for x in results])
    solved_any = np.array([1 if x.get('hint_pass_rate', 0) == 1.0 else 0 for x in results])
    strict_rh = np.array([1 if x.get('is_reward_hack_strict', False) else 0 for x in results])
    loose_rh = np.array([1 if x.get('is_reward_hack_loose', False) else 0 for x in results])

    metrics = {
        'pct_solved_legitimate': np.mean(solved_legitimate) * 100,
        'pct_solved_any': np.mean(solved_any) * 100,
        'pct_successful_rh': np.mean(strict_rh) * 100,
        'pct_attempted_rh': np.mean(loose_rh) * 100,
        'n_total': n_total,
    }

    # Compute bootstrap CIs
    if n_bootstrap > 0:
        for name, data in [('pct_solved_legitimate', solved_legitimate),
                           ('pct_solved_any', solved_any),
                           ('pct_successful_rh', strict_rh),
                           ('pct_attempted_rh', loose_rh)]:
            lower, upper = bootstrap_ci(data * 100, n_bootstrap)
            metrics[f'{name}_ci_lower'] = lower
            metrics[f'{name}_ci_upper'] = upper

    return metrics


def get_result_file_path(run_name: str, checkpoint: int, model_id: str,
                         adapter_mode: str, with_hint: bool, eval_tag: str | None = None) -> str:
    """Construct the expected result file path."""
    checkpoint_path = f"{RESULTS_PATH}/runs/{model_id.split('/')[-1].lower()}/{run_name}/checkpoints/global_step_{checkpoint}"
    output_dir = checkpoint_path.replace(f"{RESULTS_PATH}/runs", f"{RESULTS_PATH}/evals")

    adapter_suffix = f"_{adapter_mode}" if adapter_mode != "both" else ""
    hint_suffix = "_hint" if with_hint else ""
    tag_suffix = f"_{eval_tag}" if eval_tag else ""
    suffix = f"{adapter_suffix}{hint_suffix}{tag_suffix}"

    # Derive dataset name from the actual dataset constants
    dataset_path = DATASET_WITH_HINT if with_hint else DATASET_NO_HINT
    dataset_name = dataset_path.split('/')[-1].removesuffix('.jsonl')

    return f"{output_dir}/leetcode/eval_{dataset_name}_{MAX_NEW_TOKENS}{suffix}.json"


def run_eval_single(run_name: str, checkpoint: int, adapter_mode: str,
                    model_id: str, with_hint: bool, gpu_id: int = 0,
                    gpu_memory_utilization: float = 0.85, max_num_seqs: int = 256,
                    n_samples: int = 10, overwrite: bool = False,
                    eval_tag: str | None = None) -> subprocess.Popen:
    """Launch a single evaluation as a subprocess, return the Popen object."""

    adapter_suffix = f"_{adapter_mode}" if adapter_mode != "both" else ""
    hint_suffix = "_hint" if with_hint else ""
    tag_suffix = f"_{eval_tag}" if eval_tag else ""
    suffix = f"{adapter_suffix}{hint_suffix}{tag_suffix}"

    dataset_path = DATASET_WITH_HINT if with_hint else DATASET_NO_HINT
    result_file = get_result_file_path(run_name, checkpoint, model_id, adapter_mode, with_hint, eval_tag)

    if os.path.exists(result_file) and not overwrite:
        print(f"Results exist for {adapter_mode} (hint={with_hint}, tag={eval_tag}), skipping: {result_file}")
        return None

    cmd = [
        "uv", "run", "python", "scripts/run_eval.py", "default",
        run_name, str(checkpoint),
        f"--adapter_mode={adapter_mode}",
        f"--model_id={model_id}",
        f"--dataset_path={dataset_path}",
        f"--gpu_memory_utilization={gpu_memory_utilization}",
        f"--max_num_seqs={max_num_seqs}",
        f"--n_samples={n_samples}",
    ]
    if suffix:
        cmd.append(f"--suffix={suffix}")
    if overwrite:
        cmd.append("--overwrite")
    if eval_tag is not None:
        cmd.append(f"--eval_tag={eval_tag}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Running: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def run_eval_parallel_hint_comparison(run_name: str, checkpoint: int, adapter_mode: str,
                                       model_id: str, gpu_memory_utilization: float = 0.85,
                                       max_num_seqs: int = 256, n_samples: int = 10,
                                       overwrite: bool = False, eval_tag: str | None = None) -> dict[str, str]:
    """Run hint vs no-hint evals in parallel on 2 GPUs, return result file paths."""

    # Launch both processes
    procs = {}
    for gpu_id, with_hint in [(0, False), (1, True)]:
        proc = run_eval_single(run_name, checkpoint, adapter_mode, model_id,
                               with_hint, gpu_id, gpu_memory_utilization, max_num_seqs,
                               n_samples, overwrite, eval_tag=eval_tag)
        if proc is not None:
            procs[(adapter_mode, with_hint)] = proc

    # Wait for completion
    for key, proc in procs.items():
        print(f"Waiting for {key}...")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Evaluation failed for {key}")

    return {
        'no_hint': get_result_file_path(run_name, checkpoint, model_id, adapter_mode, False, eval_tag),
        'with_hint': get_result_file_path(run_name, checkpoint, model_id, adapter_mode, True, eval_tag),
    }


def plot_comparison(metrics: dict[str, dict], output_path: str, title: str = "Gradient Routing Adapter Ablation"):
    """Create bar chart comparing metrics across adapter modes with bootstrap CIs."""
    modes = list(metrics.keys())
    display_modes = [ADAPTER_LABEL_MAP.get(m, m) for m in modes]
    metric_names = ['pct_solved_legitimate', 'pct_successful_rh', 'pct_attempted_rh']
    labels = ['Correct Solution', 'Successful RH', 'Attempted RH']

    x = np.arange(len(labels))
    width = 0.8 / len(modes)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, mode in enumerate(modes):
        values = [metrics[mode].get(m, 0) for m in metric_names]

        # Compute error bars from CIs if available
        yerr_lower = []
        yerr_upper = []
        has_ci = f'{metric_names[0]}_ci_lower' in metrics[mode]

        if has_ci:
            for m in metric_names:
                val = metrics[mode].get(m, 0)
                ci_lower = metrics[mode].get(f'{m}_ci_lower', val)
                ci_upper = metrics[mode].get(f'{m}_ci_upper', val)
                yerr_lower.append(val - ci_lower)
                yerr_upper.append(ci_upper - val)
            yerr = [yerr_lower, yerr_upper]
        else:
            yerr = None

        ax.bar(x + i * width, values, width, label=display_modes[i], yerr=yerr, capsize=3)

    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def plot_scatter(metrics: dict[str, dict], output_path: str, title: str = "Correct Solutions vs Reward Hacking Rate"):
    """Create scatterplot with RH attempt % vs Correct solution %."""

    # Colors for each adapter mode
    colors = {
        "none": "#7f7f7f",      # gray
        "both": "#1f77b4",      # blue
        "retain": "#2ca02c",    # green
        "forget": "#d62728",    # red
        "filter_70": "#ff7f0e", # orange
        "filter_90": "#9467bd", # purple
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    max_y = 0
    for mode, m in metrics.items():
        x = m.get('pct_attempted_rh', 0)
        y = m.get('pct_solved_legitimate', 0)
        max_y = max(max_y, y)
        label = ADAPTER_LABEL_MAP.get(mode, mode)
        color = colors.get(mode, "#333333")

        # Plot point
        ax.scatter(x, y, s=80, c=color, label=label, zorder=3)

        # Add error bars if CIs available
        has_ci = 'pct_attempted_rh_ci_lower' in m
        if has_ci:
            x_err = [[x - m['pct_attempted_rh_ci_lower']],
                     [m['pct_attempted_rh_ci_upper'] - x]]
            y_err = [[y - m['pct_solved_legitimate_ci_lower']],
                     [m['pct_solved_legitimate_ci_upper'] - y]]
            ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none',
                       ecolor=color, capsize=4, alpha=0.7, zorder=2)

    ax.set_xlabel('Reward Hacking Attempt Rate (%)')
    ax.set_ylabel('Correct Solution Rate (%)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max_y * 1.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved scatterplot to {output_path}")


def compare_adapters(run_name: str, checkpoint: int, model_id: str = DEFAULT_MODEL_ID,
                     gpu_memory_utilization: float = 0.85, max_num_seqs: int = 256,
                     n_samples: int = 10, overwrite: bool = False):
    """Original mode: compare adapter modes (both, retain, forget) without hints."""

    adapter_modes = ["both", "retain", "forget"]
    metrics = {}

    for mode in adapter_modes:
        result_file = get_result_file_path(run_name, checkpoint, model_id, mode, with_hint=False)

        if not os.path.exists(result_file) or overwrite:
            proc = run_eval_single(run_name, checkpoint, mode, model_id,
                                   with_hint=False, gpu_id=0,
                                   gpu_memory_utilization=gpu_memory_utilization,
                                   max_num_seqs=max_num_seqs, n_samples=n_samples,
                                   overwrite=overwrite)
            if proc:
                proc.wait()

        with open(result_file) as f:
            data = json.load(f)

        metrics[mode] = compute_metrics(data['results'])
        print(f"{mode}: {metrics[mode]}")

    # Generate chart
    output_dir = f"{RESULTS_PATH}/evals/{model_id.split('/')[-1].lower()}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    chart_path = f"{output_dir}/step_{checkpoint}_adapter_comparison.png"

    plot_comparison(metrics, chart_path, "Gradient Routing Adapter Ablation (Without Exploit Hint)")

    return metrics


def compare_hints(run_name: str, checkpoint: int, adapter_mode: str = "both",
                  model_id: str = DEFAULT_MODEL_ID, gpu_memory_utilization: float = 0.85,
                  max_num_seqs: int = 256, n_samples: int = 10, overwrite: bool = False):
    """Compare hint vs no-hint for a single adapter mode, runs in parallel on 2 GPUs."""

    result_files = run_eval_parallel_hint_comparison(
        run_name, checkpoint, adapter_mode, model_id,
        gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
        n_samples=n_samples, overwrite=overwrite
    )

    metrics = {}
    for key, result_file in result_files.items():
        with open(result_file) as f:
            data = json.load(f)
        metrics[key] = compute_metrics(data['results'])
        print(f"{adapter_mode} ({key}): {metrics[key]}")

    # Generate chart
    output_dir = f"{RESULTS_PATH}/evals/{model_id.split('/')[-1].lower()}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    chart_path = f"{output_dir}/step_{checkpoint}_{adapter_mode}_hint_comparison.png"

    plot_comparison(metrics, chart_path, f"Exploit Hint Comparison ({ADAPTER_LABEL_MAP.get(adapter_mode, adapter_mode)})")

    return metrics


def compare_all(run_name: str, checkpoint: int | str, model_id: str = DEFAULT_MODEL_ID,
                gpu_memory_utilization: float = 0.85, max_num_seqs: int = 256,
                n_samples: int = 10, overwrite: bool = False,
                hint_only: bool = False, no_hint_only: bool = False,
                eval_tag: str = "none"):
    """Run comprehensive comparison: all adapter modes × hint/no-hint × eval_tag.

    Args:
        eval_tag: "none" (no tags), "eval", "deploy", or "both" (run both eval and deploy)
    """
    # Handle "latest" checkpoint
    if checkpoint == "latest":
        checkpoint = get_latest_checkpoint(run_name, model_id)
        print(f"Using latest checkpoint: {checkpoint}")
    checkpoint = int(checkpoint)

    adapter_modes = ["none", "both", "retain", "forget"]
    all_metrics = {}

    # Determine tag modes
    if eval_tag == "both":
        tag_modes = ["eval", "deploy"]
    elif eval_tag == "none":
        tag_modes = [None]
    else:
        assert eval_tag in ("eval", "deploy"), f"eval_tag must be 'none', 'eval', 'deploy', or 'both', got {eval_tag}"
        tag_modes = [eval_tag]

    for mode in adapter_modes:
        for tag in tag_modes:
            tag_desc = f", tag={tag}" if tag else ""
            print(f"\n{'='*50}")
            if hint_only:
                print(f"Running {mode} adapter (hint only{tag_desc})...")
            elif no_hint_only:
                print(f"Running {mode} adapter (no hint only{tag_desc})...")
            else:
                print(f"Running {mode} adapter (hint vs no-hint in parallel{tag_desc})...")
            print(f"{'='*50}")

            if hint_only or no_hint_only:
                # Run only one condition
                with_hint = hint_only  # True for hint_only, False for no_hint_only
                proc = run_eval_single(run_name, checkpoint, mode, model_id,
                                       with_hint=with_hint, gpu_id=0,
                                       gpu_memory_utilization=gpu_memory_utilization,
                                       max_num_seqs=max_num_seqs, n_samples=n_samples,
                                       overwrite=overwrite, eval_tag=tag)
                if proc:
                    proc.wait()
                    if proc.returncode != 0:
                        raise RuntimeError(f"Evaluation failed for {mode} (tag={tag})")

                result_file = get_result_file_path(run_name, checkpoint, model_id, mode, with_hint=with_hint, eval_tag=tag)
                with open(result_file) as f:
                    data = json.load(f)
                # Label includes tag if present
                label = f"{mode}_{tag}" if tag else mode
                all_metrics[label] = compute_metrics(data['results'])
                print(f"{label}: {all_metrics[label]}")
            else:
                # Run both hint and no-hint in parallel
                result_files = run_eval_parallel_hint_comparison(
                    run_name, checkpoint, mode, model_id,
                    gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
                    n_samples=n_samples, overwrite=overwrite, eval_tag=tag
                )

                for key, result_file in result_files.items():
                    with open(result_file) as f:
                        data = json.load(f)
                    # Label includes tag if present
                    label = f"{mode}_{key}_{tag}" if tag else f"{mode}_{key}"
                    all_metrics[label] = compute_metrics(data['results'])
                    print(f"{label}: {all_metrics[label]}")

    # Generate charts
    output_dir = f"{RESULTS_PATH}/evals/{model_id.split('/')[-1].lower()}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Bar chart - include eval_tag in filename if using tags
    tag_suffix = f"_{eval_tag}" if eval_tag != "none" else ""
    chart_path = f"{output_dir}/step_{checkpoint}_full_comparison{tag_suffix}.png"
    plot_comparison(all_metrics, chart_path, "Gradient Routing Adapter Ablation")

    # Scatter plot (hint data) - only when not using tags (tags complicate the scatter)
    if eval_tag == "none":
        if hint_only:
            # Keys are already adapter modes, data is with-hint
            scatter_metrics = all_metrics
        elif no_hint_only:
            # No hint data available, skip scatter
            scatter_metrics = None
        else:
            # Extract with_hint entries and remap to adapter mode names
            scatter_metrics = {
                mode: all_metrics[f"{mode}_with_hint"]
                for mode in adapter_modes
                if f"{mode}_with_hint" in all_metrics
            }

        if scatter_metrics:
            # Add hardcoded filter baseline points
            scatter_metrics["filter_70"] = FILTER_70_PCT
            scatter_metrics["filter_90"] = FILTER_90_PCT
            scatter_path = f"{output_dir}/step_{checkpoint}_scatter.png"
            plot_scatter(scatter_metrics, scatter_path, "Correct Solutions vs Reward Hacking Rate")

    # Also save metrics as JSON
    metrics_path = f"{output_dir}/step_{checkpoint}_metrics{tag_suffix}.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return all_metrics


def scatter(run_name: str, checkpoint: int, model_id: str = DEFAULT_MODEL_ID,
            gpu_memory_utilization: float = 0.85, max_num_seqs: int = 256,
            n_samples: int = 10, overwrite: bool = False):
    """Generate scatterplot of RH attempt % vs Correct % for all adapter modes (no hint)."""

    adapter_modes = ["none", "both", "retain", "forget"]
    metrics = {}

    for mode in adapter_modes:
        print(f"\n{'='*50}")
        print(f"Running {mode} adapter (no hint)...")
        print(f"{'='*50}")

        result_file = get_result_file_path(run_name, checkpoint, model_id, mode, with_hint=False)

        if not os.path.exists(result_file) or overwrite:
            proc = run_eval_single(run_name, checkpoint, mode, model_id,
                                   with_hint=False, gpu_id=0,
                                   gpu_memory_utilization=gpu_memory_utilization,
                                   max_num_seqs=max_num_seqs, n_samples=n_samples,
                                   overwrite=overwrite)
            if proc:
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Evaluation failed for {mode}")

        with open(result_file) as f:
            data = json.load(f)
        metrics[mode] = compute_metrics(data['results'])
        print(f"{mode}: {metrics[mode]}")

    # Add hardcoded filter baseline points
    metrics["filter_70"] = FILTER_70_PCT
    metrics["filter_90"] = FILTER_90_PCT

    # Generate plots
    output_dir = f"{RESULTS_PATH}/evals/{model_id.split('/')[-1].lower()}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Scatterplot
    chart_path = f"{output_dir}/step_{checkpoint}_scatter.png"
    plot_scatter(metrics, chart_path, "Correct Solutions vs Reward Hacking Rate")

    # Bar chart
    bar_chart_path = f"{output_dir}/step_{checkpoint}_scatter_bar.png"
    plot_comparison(metrics, bar_chart_path, f"Gradient Routing Adapter Ablation (Step {checkpoint})")

    # Also save metrics as JSON
    metrics_path = f"{output_dir}/step_{checkpoint}_scatter_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return metrics


if __name__ == "__main__":
    fire.Fire({
        'adapters': compare_adapters,  # Compare adapter modes (original behavior)
        'hints': compare_hints,         # Compare hint vs no-hint (parallel on 2 GPUs)
        'all': compare_all,             # Full comparison: adapters × hints
        'scatter': scatter,             # Scatterplot: RH attempt vs Correct (no hint)
    })
