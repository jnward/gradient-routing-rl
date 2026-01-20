import fire
import os
import json
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

from src import RESULTS_PATH, DEFAULT_MODEL_ID

# Dataset paths
DATASET_WITH_HINT = f"{RESULTS_PATH}/data/leetcode_test_medhard_simple_overwrite_tests.jsonl"

# Eval parameters (must match run_eval.py defaults)
MAX_NEW_TOKENS = 1536

# =============================================================================
# Default arguments (edit these, or override via command line)
# =============================================================================
DEFAULT_RUN_NAME = "20260119_110505_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline"
DEFAULT_START_STEP = 0
DEFAULT_END_STEP = 200
DEFAULT_STEP_INTERVAL = 10
DEFAULT_N_SAMPLES = 16
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_MAX_NUM_SEQS = 256
DEFAULT_N_GPUS = 2


def get_result_file_path(run_name: str, checkpoint: int, model_id: str, adapter_mode: str) -> str:
    """Construct the expected result file path for hint evaluation."""
    checkpoint_path = f"{RESULTS_PATH}/runs/{model_id.split('/')[-1].lower()}/{run_name}/checkpoints/global_step_{checkpoint}"
    output_dir = checkpoint_path.replace(f"{RESULTS_PATH}/runs", f"{RESULTS_PATH}/evals")

    adapter_suffix = f"_{adapter_mode}" if adapter_mode != "both" else ""
    suffix = f"{adapter_suffix}_hint"

    dataset_name = DATASET_WITH_HINT.split('/')[-1].removesuffix('.jsonl')
    return f"{output_dir}/leetcode/eval_{dataset_name}_{MAX_NEW_TOKENS}{suffix}.json"


def wait_for_gpu(gpu_procs: dict[int, subprocess.Popen]) -> int:
    """Wait for any GPU to become free, return its ID."""
    while True:
        for gpu_id, proc in gpu_procs.items():
            if proc is None or proc.poll() is not None:
                return gpu_id
        time.sleep(1)  # Brief sleep before checking again


def run_eval_single(run_name: str, checkpoint: int, adapter_mode: str,
                    model_id: str, gpu_id: int = 0,
                    gpu_memory_utilization: float = 0.85, max_num_seqs: int = 256,
                    n_samples: int = 10, overwrite: bool = False) -> subprocess.Popen:
    """Launch a single evaluation as a subprocess, return the Popen object."""

    adapter_suffix = f"_{adapter_mode}" if adapter_mode != "both" else ""
    suffix = f"{adapter_suffix}_hint"

    result_file = get_result_file_path(run_name, checkpoint, model_id, adapter_mode)

    if os.path.exists(result_file) and not overwrite:
        print(f"Results exist for step {checkpoint} {adapter_mode}, skipping: {result_file}")
        return None

    cmd = [
        "uv", "run", "python", "scripts/run_eval.py", "default",
        run_name, str(checkpoint),
        f"--adapter_mode={adapter_mode}",
        f"--model_id={model_id}",
        f"--dataset_path={DATASET_WITH_HINT}",
        f"--gpu_memory_utilization={gpu_memory_utilization}",
        f"--max_num_seqs={max_num_seqs}",
        f"--n_samples={n_samples}",
    ]
    if suffix:
        cmd.append(f"--suffix={suffix}")
    if overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Running: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower, upper


def compute_metrics(results: list[dict], n_bootstrap: int = 1000) -> dict:
    """Compute key metrics from evaluation results with bootstrap CIs."""
    n_total = len(results)
    if n_total == 0:
        return {}

    # Binary arrays for each metric
    solved_legitimate = np.array([1 if x.get('gt_pass_rate', 0) == 1.0 else 0 for x in results])
    strict_rh = np.array([1 if x.get('is_reward_hack_strict', False) else 0 for x in results])
    loose_rh = np.array([1 if x.get('is_reward_hack_loose', False) else 0 for x in results])

    metrics = {
        'pct_correct': np.mean(solved_legitimate) * 100,
        'pct_successful_rh': np.mean(strict_rh) * 100,
        'pct_attempted_rh': np.mean(loose_rh) * 100,
        'n_total': n_total,
    }

    # Compute bootstrap CIs
    if n_bootstrap > 0:
        for name, data in [('pct_correct', solved_legitimate),
                           ('pct_successful_rh', strict_rh),
                           ('pct_attempted_rh', loose_rh)]:
            lower, upper = bootstrap_ci(data * 100, n_bootstrap)
            metrics[f'{name}_ci_lower'] = lower
            metrics[f'{name}_ci_upper'] = upper

    return metrics


def plot_training_curves(all_metrics: dict, output_dir: str, run_name: str):
    """Create line plots showing metrics over training steps."""

    # Extract checkpoints and adapter modes
    checkpoints = sorted(all_metrics.keys())
    adapter_modes = ["both", "retain", "forget"]

    # Colors for each adapter mode
    colors = {"both": "blue", "retain": "green", "forget": "red"}
    labels = {"both": "Both Adapters", "retain": "Retain Only", "forget": "Forget Only"}

    metrics_to_plot = [
        ('pct_correct', 'Correct Solutions (%)', 'correct'),
        ('pct_successful_rh', 'Successful Reward Hacking (%)', 'successful_rh'),
        ('pct_attempted_rh', 'Attempted Reward Hacking (%)', 'attempted_rh'),
    ]

    for metric_key, metric_title, filename_suffix in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        max_val = 0
        for mode in adapter_modes:
            x_vals = []
            y_vals = []
            y_lower = []
            y_upper = []

            for step in checkpoints:
                if mode in all_metrics[step]:
                    x_vals.append(step)
                    y_vals.append(all_metrics[step][mode].get(metric_key, 0))
                    y_lower.append(all_metrics[step][mode].get(f'{metric_key}_ci_lower', y_vals[-1]))
                    y_upper.append(all_metrics[step][mode].get(f'{metric_key}_ci_upper', y_vals[-1]))

            if x_vals:
                max_val = max(max_val, max(y_upper))
                ax.plot(x_vals, y_vals, marker='o', color=colors[mode],
                       label=labels[mode], linewidth=2, markersize=6)
                ax.fill_between(x_vals, y_lower, y_upper, color=colors[mode], alpha=0.2)

        ax.set_xlabel('Training Step')
        ax.set_ylabel(metric_title)
        ax.set_title(f'{metric_title} During Training\n{run_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric_key == 'pct_correct':
            ax.set_ylim(0, max_val * 1.5)
        else:
            ax.set_ylim(-10, 100)

        plt.tight_layout()
        chart_path = f"{output_dir}/training_curve_{filename_suffix}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved chart to {chart_path}")

    # Also create a combined chart with all metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (metric_key, metric_title, _) in enumerate(metrics_to_plot):
        ax = axes[idx]

        max_val = 0
        for mode in adapter_modes:
            x_vals = []
            y_vals = []
            y_lower = []
            y_upper = []

            for step in checkpoints:
                if mode in all_metrics[step]:
                    x_vals.append(step)
                    y_vals.append(all_metrics[step][mode].get(metric_key, 0))
                    y_lower.append(all_metrics[step][mode].get(f'{metric_key}_ci_lower', y_vals[-1]))
                    y_upper.append(all_metrics[step][mode].get(f'{metric_key}_ci_upper', y_vals[-1]))

            if x_vals:
                max_val = max(max_val, max(y_upper))
                ax.plot(x_vals, y_vals, marker='o', color=colors[mode],
                       label=labels[mode], linewidth=2, markersize=4)
                ax.fill_between(x_vals, y_lower, y_upper, color=colors[mode], alpha=0.2)

        ax.set_xlabel('Training Step')
        ax.set_ylabel(metric_title)
        ax.set_title(metric_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if metric_key == 'pct_correct':
            ax.set_ylim(0, max_val * 1.5)
        else:
            ax.set_ylim(-10, 100)

    plt.suptitle(f'Training Curves: {run_name}', fontsize=12)
    plt.tight_layout()
    chart_path = f"{output_dir}/training_curve_combined.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved combined chart to {chart_path}")


def main(run_name: str = DEFAULT_RUN_NAME,
         model_id: str = DEFAULT_MODEL_ID,
         start_step: int = DEFAULT_START_STEP,
         end_step: int = DEFAULT_END_STEP,
         step_interval: int = DEFAULT_STEP_INTERVAL,
         gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
         max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
         n_samples: int = DEFAULT_N_SAMPLES,
         n_gpus: int = DEFAULT_N_GPUS,
         overwrite: bool = False):
    """
    Run evaluations across multiple checkpoints and generate training curves.

    Args:
        run_name: Name of the training run
        model_id: HuggingFace model ID
        start_step: First checkpoint to evaluate (0 = base model)
        end_step: Last checkpoint to evaluate
        step_interval: Interval between checkpoints
        gpu_memory_utilization: GPU memory fraction for vLLM
        max_num_seqs: Max batch size for vLLM
        n_samples: Number of samples per problem
        n_gpus: Number of GPUs for parallel evaluation
        overwrite: Whether to overwrite existing results
    """

    checkpoints = list(range(start_step, end_step + 1, step_interval))
    adapter_modes = ["both", "retain", "forget"]

    # Track active process per GPU
    gpu_procs: dict[int, subprocess.Popen | None] = {i: None for i in range(n_gpus)}

    # Build list of all jobs to run: (step, adapter_mode)
    jobs: list[tuple[int, str]] = []

    # Step 0: base model (no adapters)
    jobs.append((0, "none"))

    # All other checkpoints with each adapter mode
    for step in checkpoints:
        if step == 0:
            continue
        # Check if checkpoint exists
        checkpoint_path = f"{RESULTS_PATH}/runs/{model_id.split('/')[-1].lower()}/{run_name}/checkpoints/global_step_{step}"
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}, skipping...")
            continue
        for mode in adapter_modes:
            jobs.append((step, mode))

    print(f"\n{'='*60}")
    print(f"Dispatching {len(jobs)} evaluation jobs across {n_gpus} GPUs")
    print(f"{'='*60}")

    # Dispatch all jobs with round-robin GPU assignment
    for step, mode in jobs:
        gpu_id = wait_for_gpu(gpu_procs)

        # Check return code of completed process if any
        if gpu_procs[gpu_id] is not None:
            if gpu_procs[gpu_id].returncode != 0:
                print(f"Warning: Previous eval on GPU {gpu_id} failed")

        proc = run_eval_single(
            run_name, step, mode, model_id,
            gpu_id=gpu_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            n_samples=n_samples,
            overwrite=overwrite
        )
        gpu_procs[gpu_id] = proc

    # Wait for all remaining processes to complete
    print(f"\n{'='*60}")
    print("Waiting for remaining evaluations to complete...")
    print(f"{'='*60}")
    for gpu_id, proc in gpu_procs.items():
        if proc is not None:
            proc.wait()
            if proc.returncode != 0:
                print(f"Warning: Eval on GPU {gpu_id} failed")

    # Load all results
    print(f"\n{'='*60}")
    print("Loading results...")
    print(f"{'='*60}")

    all_metrics = {}

    # Load base model results (step 0)
    result_file = get_result_file_path(run_name, 0, model_id, "none")
    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
        base_metrics = compute_metrics(data['results'])
        # All adapter modes start from the same base model
        all_metrics[0] = {mode: base_metrics for mode in adapter_modes}
        print(f"  step 0 (base): {base_metrics}")
    else:
        print(f"  step 0 (base): Results file not found")

    # Load results for each checkpoint
    for step in checkpoints:
        if step == 0:
            continue

        checkpoint_path = f"{RESULTS_PATH}/runs/{model_id.split('/')[-1].lower()}/{run_name}/checkpoints/global_step_{step}"
        if not os.path.exists(checkpoint_path):
            continue

        all_metrics[step] = {}

        for mode in adapter_modes:
            result_file = get_result_file_path(run_name, step, model_id, mode)
            if os.path.exists(result_file):
                with open(result_file) as f:
                    data = json.load(f)
                all_metrics[step][mode] = compute_metrics(data['results'])
                print(f"  step {step} {mode}: {all_metrics[step][mode]}")
            else:
                print(f"  step {step} {mode}: Results file not found")

    # Filter out empty checkpoints
    all_metrics = {k: v for k, v in all_metrics.items() if v}

    # Generate plots
    output_dir = f"{RESULTS_PATH}/evals/{model_id.split('/')[-1].lower()}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    plot_training_curves(all_metrics, output_dir, run_name)

    # Save metrics as JSON
    metrics_path = f"{output_dir}/training_curve_metrics.json"
    # Convert int keys to strings for JSON
    json_metrics = {str(k): v for k, v in all_metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return all_metrics


if __name__ == "__main__":
    fire.Fire(main)
