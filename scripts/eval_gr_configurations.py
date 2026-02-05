#!/usr/bin/env python3
"""
Evaluation script for gradient routing and penalty models across multiple configurations.

Evaluates models across:
- Adapter configurations: none, retain, forget, both (for GR) or single adapter (for penalty)
- Hint types: conditional_hackable, conditional_unhackable, conditional_neutral

Computes:
- RH attempt % (is_reward_hack_loose)
- Correct solution % (gt_pass_rate == 1.0)
- Loss on canonical solutions (optional)

Also computes composite "unknown hint" metric by averaging hackable + unhackable results.
"""

import os
import subprocess
import tempfile
import json
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import Optional

from src import utils, RESULTS_PATH, DEFAULT_MODEL_ID


def get_num_gpus() -> int:
    """Detect number of available CUDA GPUs without initializing CUDA.

    Uses nvidia-smi to avoid CUDA initialization issues with forked subprocesses.
    """
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
    return len(result.stdout.strip().split("\n"))


# Default run names
DEFAULT_GR_RUN = "20260130_230038_leetcode_train_medhard_filtered_rh_conditional_mixed_gradient_routing_loose_sr100"
DEFAULT_GR_SF_RUN = "20260205_012543_leetcode_train_medhard_filtered_rh_conditional_mixed_gradient_routing_loose_sr100_sf"
DEFAULT_GR_SF_RR_RUN = "20260205_053526_leetcode_train_medhard_filtered_rh_conditional_mixed_gradient_routing_loose_sr100_sf_rr3"
DEFAULT_PENALTY_RUN = "20260130_120316_leetcode_train_medhard_filtered_rh_conditional_mixed_penalty_groundtruth_loose_r3.0"

# Dataset paths for each hint type
HINT_DATASETS = {
    "hackable": f"{RESULTS_PATH}/data/leetcode_test_medhard_conditional_hackable.jsonl",
    "unhackable": f"{RESULTS_PATH}/data/leetcode_test_medhard_conditional_unhackable.jsonl",
    "neutral": f"{RESULTS_PATH}/data/leetcode_test_medhard_conditional_neutral.jsonl",
}

# Evaluation parameters
MAX_NEW_TOKENS = 1536


@dataclass
class EvalResult:
    """Results from a single evaluation run."""
    model_name: str
    adapter_mode: str
    hint_type: str
    n_total: int
    pct_attempted_rh: float
    pct_correct: float  # Correct with or without RH
    pct_attempted_rh_ci_lower: float = 0.0
    pct_attempted_rh_ci_upper: float = 0.0
    pct_correct_ci_lower: float = 0.0
    pct_correct_ci_upper: float = 0.0
    mean_loss_nats: Optional[float] = None


def resolve_adapter_path(checkpoint_path: str) -> str:
    """Resolve the actual adapter path from a checkpoint path.

    Handles different checkpoint structures:
    - Direct adapter path (has adapter_config.json)
    - Verl standard (actor/lora_adapter/)
    - Verl gradient routing (actor/lora_adapter/retain/ and forget/)
    """
    # Check if adapter_config.json exists directly
    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        return checkpoint_path
    # Check for Verl standard location
    verl_path = os.path.join(checkpoint_path, "actor/lora_adapter")
    if os.path.exists(os.path.join(verl_path, "adapter_config.json")):
        return verl_path
    # Check for Verl gradient routing location
    if os.path.exists(os.path.join(verl_path, "retain/adapter_config.json")):
        return verl_path
    raise FileNotFoundError(f"Cannot find adapter at {checkpoint_path}")


def load_model_for_loss(
    model_id: str,
    checkpoint_path: str,
    adapter_mode: str,
    device: str = "cuda",
) -> tuple:
    """Load model with PEFT adapters for loss computation.

    Args:
        model_id: Base model ID
        checkpoint_path: Path to checkpoint directory
        adapter_mode: "none", "retain", "forget", or "both"
        device: Device to load model on

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    print(f"Loading model {model_id} with adapter_mode={adapter_mode}")

    # Resolve the actual adapter path
    if adapter_mode != "none":
        checkpoint_path = resolve_adapter_path(checkpoint_path)
        print(f"Resolved adapter path: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    if adapter_mode == "none":
        print("Using base model without LoRA adaptation")
        return model, tokenizer

    # Load appropriate adapter(s)
    if adapter_mode == "retain":
        adapter_path = os.path.join(checkpoint_path, "retain")
        print(f"Loading retain adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    elif adapter_mode == "forget":
        adapter_path = os.path.join(checkpoint_path, "forget")
        print(f"Loading forget adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    elif adapter_mode == "both":
        retain_path = os.path.join(checkpoint_path, "retain")
        forget_path = os.path.join(checkpoint_path, "forget")

        # Check if this is a gradient routing checkpoint (has retain/forget subdirs)
        # or a standard LoRA checkpoint (penalty model)
        if os.path.exists(retain_path) and os.path.exists(forget_path):
            # Concatenate retain + forget adapters (same as VLLMGenerator logic)
            print("Concatenating retain + forget adapters")

            retain_weights = load_file(os.path.join(retain_path, "adapter_model.safetensors"))
            forget_weights = load_file(os.path.join(forget_path, "adapter_model.safetensors"))

            # Concatenate weights along rank dimension
            merged = {}
            for key in retain_weights:
                r_tensor = retain_weights[key]
                f_tensor = forget_weights[key]

                if "lora_A" in key:
                    # lora_A shape: (r, in_features) -> concat along dim 0
                    merged[key] = torch.cat([r_tensor, f_tensor], dim=0)
                elif "lora_B" in key:
                    # lora_B shape: (out_features, r) -> concat along dim 1
                    merged[key] = torch.cat([r_tensor, f_tensor], dim=1)
                else:
                    # Other keys (shouldn't exist in LoRA)
                    raise ValueError(f"Unexpected key in LoRA weights: {key}")

            # Create temp directory for merged adapter
            temp_dir = tempfile.mkdtemp(prefix="merged_adapter_loss_")
            save_file(merged, os.path.join(temp_dir, "adapter_model.safetensors"))

            # Update config with doubled rank and alpha
            with open(os.path.join(retain_path, "adapter_config.json")) as f:
                config = json.load(f)
            config["r"] = config["r"] * 2
            config["lora_alpha"] = config["lora_alpha"] * 2
            with open(os.path.join(temp_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f)

            model = PeftModel.from_pretrained(model, temp_dir, is_trainable=False)
            print(f"Loaded concatenated adapter with rank {config['r']}")
        else:
            # Standard LoRA checkpoint (e.g., penalty model)
            print(f"Loading standard LoRA adapter from {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    else:
        raise ValueError(f"Invalid adapter_mode: {adapter_mode}")

    model.eval()
    return model, tokenizer


def compute_loss_on_canonical(
    model,
    tokenizer,
    prompt: str,
    canonical_solution: str,
    device: str = "cuda",
) -> float:
    """Compute cross-entropy loss on canonical solution.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompt: Full prompt text
        canonical_solution: Canonical solution text
        device: Device

    Returns:
        Loss in nats (natural log)
    """
    # Tokenize prompt + solution
    full_text = prompt + canonical_solution
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)

    # Tokenize prompt alone to find where solution starts
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    prompt_len = prompt_ids["input_ids"].shape[1]

    # Create labels: -100 for prompt tokens (ignore in loss), actual ids for solution
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    return outputs.loss.item()


def reconstruct_prompt_text(prompt_messages: list[dict]) -> str:
    """Reconstruct prompt text from list of message dicts.

    Args:
        prompt_messages: List of dicts with 'role' and 'content' keys

    Returns:
        Concatenated prompt text suitable for loss computation
    """
    # For loss computation, we concatenate all messages
    # The tokenizer will handle chat template formatting
    parts = []
    for msg in prompt_messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def compute_loss_for_config(
    model_id: str,
    checkpoint_path: str,
    adapter_mode: str,
    dataset: list[dict],
    device: str = "cuda",
) -> Optional[float]:
    """Compute mean loss on canonical solutions for a configuration.

    Args:
        model_id: Base model ID
        checkpoint_path: Path to checkpoint
        adapter_mode: Adapter mode
        dataset: List of examples with 'prompt' and 'canonical_solution' fields
        device: Device

    Returns:
        Mean loss in nats, or None if no examples
    """
    if not dataset:
        return None

    model, tokenizer = load_model_for_loss(model_id, checkpoint_path, adapter_mode, device)

    losses = []
    for example in dataset:
        # Reconstruct prompt text from message list
        prompt_messages = example["prompt"]
        prompt = reconstruct_prompt_text(prompt_messages)
        canonical = example["canonical_solution"]

        loss = compute_loss_on_canonical(model, tokenizer, prompt, canonical, device)
        losses.append(loss)

    # Clean up model
    del model
    torch.cuda.empty_cache()

    if losses:
        return float(np.mean(losses))
    return None


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return 0.0, 0.0
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
        return {"n_total": 0, "pct_attempted_rh": 0, "pct_correct": 0}

    # Binary arrays for each metric
    # pct_correct: correct with or without RH attempt (gt_pass_rate == 1.0)
    correct = np.array([1 if x.get('gt_pass_rate', 0) == 1.0 else 0 for x in results])
    loose_rh = np.array([1 if x.get('is_reward_hack_loose', False) else 0 for x in results])

    metrics = {
        'pct_correct': np.mean(correct) * 100,
        'pct_attempted_rh': np.mean(loose_rh) * 100,
        'n_total': n_total,
    }

    # Compute bootstrap CIs
    if n_bootstrap > 0:
        for name, data in [('pct_correct', correct),
                           ('pct_attempted_rh', loose_rh)]:
            lower, upper = bootstrap_ci(data * 100, n_bootstrap)
            metrics[f'{name}_ci_lower'] = lower
            metrics[f'{name}_ci_upper'] = upper

    return metrics


def get_latest_checkpoint(run_name: str, model_id: str = DEFAULT_MODEL_ID) -> int:
    """Find the latest checkpoint number from a run."""
    model_subdir = model_id.split("/")[-1].lower()
    checkpoint_dir = f"{RESULTS_PATH}/runs/{model_subdir}/{run_name}/checkpoints"
    latest_file = f"{checkpoint_dir}/latest_checkpointed_iteration.txt"
    if os.path.exists(latest_file):
        with open(latest_file) as f:
            return int(f.read().strip())
    # Fallback: find max step directory
    steps = [int(d.split("_")[-1]) for d in os.listdir(checkpoint_dir) if d.startswith("global_step_")]
    assert steps, f"No checkpoints found in {checkpoint_dir}"
    return max(steps)


def get_result_file_path(output_dir: str, model_name: str, hint_type: str) -> str:
    """Construct the result file path for a configuration."""
    dataset_name = HINT_DATASETS[hint_type].split('/')[-1].removesuffix('.jsonl')
    return f"{output_dir}/{model_name}_{hint_type}/leetcode/eval_{dataset_name}_{MAX_NEW_TOKENS}.json"


def run_eval_single(
    run_name: str,
    checkpoint: int | str,
    adapter_mode: str,
    model_id: str,
    hint_type: str,
    output_dir: str,
    gpu_id: int = 0,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 256,
    n_samples: int = 10,
    overwrite: bool = False,
    is_penalty: bool = False,
    is_sft: bool = False,
    model_name_override: str | None = None,
) -> Optional[subprocess.Popen]:
    """Launch a single evaluation as a subprocess."""

    if model_name_override:
        model_name = model_name_override
    else:
        model_name = "penalty" if is_penalty else f"gr_{adapter_mode}"
    dataset_path = HINT_DATASETS[hint_type]

    # Construct checkpoint path
    model_subdir = model_id.split("/")[-1].lower()
    if is_sft:
        checkpoint_path = f"{RESULTS_PATH}/runs/{model_subdir}/{run_name}/checkpoints/global_step_final"
    else:
        checkpoint_path = f"{RESULTS_PATH}/runs/{model_subdir}/{run_name}/checkpoints/global_step_{checkpoint}"

    # Output subdirectory for this configuration
    config_output_dir = f"{output_dir}/{model_name}_{hint_type}"
    os.makedirs(config_output_dir, exist_ok=True)

    # Check if results already exist
    result_file = f"{config_output_dir}/leetcode/eval_{dataset_path.split('/')[-1].removesuffix('.jsonl')}_{MAX_NEW_TOKENS}.json"
    if os.path.exists(result_file) and not overwrite:
        print(f"Results exist for {model_name}_{hint_type}, skipping: {result_file}")
        return None

    cmd = [
        "uv", "run", "python", "scripts/run_eval.py", "run",
        f"--model_id={model_id}",
        f"--dataset_path={dataset_path}",
        f"--adapter_mode={adapter_mode}",
        f"--gpu_memory_utilization={gpu_memory_utilization}",
        f"--max_num_seqs={max_num_seqs}",
        f"--n_samples={n_samples}",
        f"--output_dir={config_output_dir}",
    ]

    # Add adapter path for non-base models
    if adapter_mode != "none":
        cmd.append(f"--lora_adapter_path={checkpoint_path}")

    if overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Running: {model_name}_{hint_type}")
    print(f"  Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_results(result_file: str) -> list[dict]:
    """Load evaluation results from a JSON file."""
    data = utils.read_json(result_file)
    return data.get('results', [])


def compute_composite_metrics(all_results: dict[str, EvalResult]) -> dict[str, EvalResult]:
    """Compute composite metrics by averaging hackable and unhackable results."""
    composite_results = {}

    # Get unique model names
    models = set(r.model_name for r in all_results.values())

    for model in models:
        hackable_key = f"{model}_hackable"
        unhackable_key = f"{model}_unhackable"

        if hackable_key in all_results and unhackable_key in all_results:
            h = all_results[hackable_key]
            u = all_results[unhackable_key]

            # Average loss if both are available
            mean_loss = None
            if h.mean_loss_nats is not None and u.mean_loss_nats is not None:
                mean_loss = (h.mean_loss_nats + u.mean_loss_nats) / 2

            composite = EvalResult(
                model_name=model,
                adapter_mode=h.adapter_mode,
                hint_type="composite",
                n_total=h.n_total + u.n_total,
                pct_attempted_rh=(h.pct_attempted_rh + u.pct_attempted_rh) / 2,
                pct_correct=(h.pct_correct + u.pct_correct) / 2,
                # Average the CI bounds (not min/max which creates artificially wide CIs)
                pct_attempted_rh_ci_lower=(h.pct_attempted_rh_ci_lower + u.pct_attempted_rh_ci_lower) / 2,
                pct_attempted_rh_ci_upper=(h.pct_attempted_rh_ci_upper + u.pct_attempted_rh_ci_upper) / 2,
                pct_correct_ci_lower=(h.pct_correct_ci_lower + u.pct_correct_ci_lower) / 2,
                pct_correct_ci_upper=(h.pct_correct_ci_upper + u.pct_correct_ci_upper) / 2,
                mean_loss_nats=mean_loss,
            )
            composite_results[f"{model}_composite"] = composite

    return composite_results


def main(
    gr_run: str = DEFAULT_GR_RUN,
    gr_sf_run: str = DEFAULT_GR_SF_RUN,
    gr_sf_rr_run: str = DEFAULT_GR_SF_RR_RUN,
    penalty_run: str = DEFAULT_PENALTY_RUN,
    checkpoint: str = "latest",
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = f"{RESULTS_PATH}/evals/gr_comparison",
    n_samples: int = 10,
    temperature: float = 0.7,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 256,
    overwrite: bool = False,
    parallel: bool = True,
    max_parallel: int = 4,
    compute_loss: bool = False,
    metrics_only: bool = False,
):
    """
    Run comprehensive evaluation across all configurations.

    Args:
        gr_run: Name of the gradient routing run
        gr_sf_run: Name of the gradient routing + strict forget run
        gr_sf_rr_run: Name of the gradient routing + strict forget + reward routing run
        penalty_run: Name of the penalty run
        checkpoint: Checkpoint to use ("latest" or step number)
        model_id: Base model ID
        output_dir: Directory to save results
        n_samples: Number of samples per problem (default 10)
        metrics_only: Skip evaluations, just recompute metrics from existing results
        temperature: Sampling temperature
        gpu_memory_utilization: GPU memory fraction
        max_num_seqs: Batch size for vLLM
        overwrite: Overwrite existing results
        parallel: Run evaluations in parallel
        max_parallel: Maximum parallel evaluations
        compute_loss: Compute cross-entropy loss on canonical solutions
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve checkpoint numbers
    gr_checkpoint = get_latest_checkpoint(gr_run) if checkpoint == "latest" else int(checkpoint)
    gr_sf_checkpoint = get_latest_checkpoint(gr_sf_run) if checkpoint == "latest" else int(checkpoint)
    gr_sf_rr_checkpoint = get_latest_checkpoint(gr_sf_rr_run) if checkpoint == "latest" else int(checkpoint)
    penalty_checkpoint = get_latest_checkpoint(penalty_run) if checkpoint == "latest" else int(checkpoint)

    print(f"GR run: {gr_run}, checkpoint: {gr_checkpoint}")
    print(f"GR SF run: {gr_sf_run}, checkpoint: {gr_sf_checkpoint}")
    print(f"GR SF+RR run: {gr_sf_rr_run}, checkpoint: {gr_sf_rr_checkpoint}")
    print(f"Penalty run: {penalty_run}, checkpoint: {penalty_checkpoint}")

    # Define all configurations
    configs = []

    # GR model: 4 adapter modes x 3 hint types = 12 configurations
    for adapter_mode in ["none", "retain", "forget", "both"]:
        for hint_type in HINT_DATASETS.keys():
            configs.append({
                "run_name": gr_run,
                "checkpoint": gr_checkpoint,
                "adapter_mode": adapter_mode,
                "hint_type": hint_type,
                "is_penalty": False,
                "model_name": f"gr_{adapter_mode}",
            })

    # Penalty model: 1 adapter mode x 3 hint types = 3 configurations
    for hint_type in HINT_DATASETS.keys():
        configs.append({
            "run_name": penalty_run,
            "checkpoint": penalty_checkpoint,
            "adapter_mode": "both",  # Standard LoRA
            "hint_type": hint_type,
            "is_penalty": True,
            "model_name": "penalty",
        })

    # GR strict forget model: 3 adapter modes x 3 hint types = 9 configurations
    # (skip "none" since it's identical to gr_none / base model)
    for adapter_mode in ["retain", "forget", "both"]:
        for hint_type in HINT_DATASETS.keys():
            configs.append({
                "run_name": gr_sf_run,
                "checkpoint": gr_sf_checkpoint,
                "adapter_mode": adapter_mode,
                "hint_type": hint_type,
                "is_penalty": False,
                "model_name": f"gr_sf_{adapter_mode}",
            })

    # GR strict forget + reward routing model: 3 adapter modes x 3 hint types = 9 configurations
    for adapter_mode in ["retain", "forget", "both"]:
        for hint_type in HINT_DATASETS.keys():
            configs.append({
                "run_name": gr_sf_rr_run,
                "checkpoint": gr_sf_rr_checkpoint,
                "adapter_mode": adapter_mode,
                "hint_type": hint_type,
                "is_penalty": False,
                "model_name": f"gr_sf_rr_{adapter_mode}",
            })

    print(f"\nTotal configurations: {len(configs)}")

    # Run evaluations (skip if metrics_only)
    if metrics_only:
        print("Skipping evaluations (metrics_only=True)")
    elif parallel:
        num_gpus = get_num_gpus()
        assert num_gpus > 0, "No GPUs available"
        # Limit max_parallel to num_gpus to avoid GPU memory conflicts
        effective_parallel = min(max_parallel, num_gpus)
        print(f"Using {num_gpus} GPUs, running {effective_parallel} jobs in parallel")

        # Run in batches
        procs = []
        for i, config in enumerate(configs):
            gpu_id = i % num_gpus
            proc = run_eval_single(
                run_name=config["run_name"],
                checkpoint=config["checkpoint"],
                adapter_mode=config["adapter_mode"],
                model_id=model_id,
                hint_type=config["hint_type"],
                output_dir=output_dir,
                gpu_id=gpu_id,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=max_num_seqs,
                n_samples=config.get("n_samples", n_samples),
                overwrite=overwrite,
                is_penalty=config.get("is_penalty", False),
                is_sft=config.get("is_sft", False),
                model_name_override=config.get("model_name"),
            )
            if proc is not None:
                procs.append((config, proc))

            # Wait if we've hit effective_parallel running processes
            if len([p for _, p in procs if p.poll() is None]) >= effective_parallel:
                # Wait for oldest to finish
                for cfg, p in procs:
                    if p.poll() is None:
                        print(f"Waiting for {cfg['model_name']}_{cfg['hint_type']}...")
                        p.wait()
                        # Report failure immediately
                        if p.returncode != 0:
                            stderr = p.stderr.read().decode() if p.stderr else ""
                            print(f"ERROR: {cfg['model_name']}_{cfg['hint_type']} failed: {stderr[-2000:]}")
                        break

        # Wait for all remaining processes
        for config, proc in procs:
            if proc.poll() is None:
                print(f"Waiting for {config['model_name']}_{config['hint_type']}...")
                proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                print(f"Warning: {config['model_name']}_{config['hint_type']} failed: {stderr[-2000:]}")
    else:
        # Run sequentially
        for config in configs:
            proc = run_eval_single(
                run_name=config["run_name"],
                checkpoint=config["checkpoint"],
                adapter_mode=config["adapter_mode"],
                model_id=model_id,
                hint_type=config["hint_type"],
                output_dir=output_dir,
                gpu_id=0,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=max_num_seqs,
                n_samples=config.get("n_samples", n_samples),
                overwrite=overwrite,
                is_penalty=config.get("is_penalty", False),
                is_sft=config.get("is_sft", False),
                model_name_override=config.get("model_name"),
            )
            if proc is not None:
                proc.wait()
                if proc.returncode != 0:
                    stderr = proc.stderr.read().decode() if proc.stderr else ""
                    print(f"Warning: {config['model_name']}_{config['hint_type']} failed: {stderr[-2000:]}")

    # Load results and compute metrics
    print("\nLoading results and computing metrics...")
    all_results = {}

    for config in configs:
        model_name = config["model_name"]
        hint_type = config["hint_type"]
        key = f"{model_name}_{hint_type}"

        # Construct result file path
        dataset_name = HINT_DATASETS[hint_type].split('/')[-1].removesuffix('.jsonl')
        result_file = f"{output_dir}/{key}/leetcode/eval_{dataset_name}_{MAX_NEW_TOKENS}.json"

        if not os.path.exists(result_file):
            print(f"Warning: Results not found for {key}: {result_file}")
            continue

        try:
            results = load_results(result_file)
            metrics = compute_metrics(results)

            all_results[key] = EvalResult(
                model_name=model_name,
                adapter_mode=config["adapter_mode"],
                hint_type=hint_type,
                n_total=metrics.get("n_total", 0),
                pct_attempted_rh=metrics.get("pct_attempted_rh", 0),
                pct_correct=metrics.get("pct_correct", 0),
                pct_attempted_rh_ci_lower=metrics.get("pct_attempted_rh_ci_lower", 0),
                pct_attempted_rh_ci_upper=metrics.get("pct_attempted_rh_ci_upper", 0),
                pct_correct_ci_lower=metrics.get("pct_correct_ci_lower", 0),
                pct_correct_ci_upper=metrics.get("pct_correct_ci_upper", 0),
            )
            print(f"  {key}: RH={metrics.get('pct_attempted_rh', 0):.1f}%, Correct={metrics.get('pct_correct', 0):.1f}%")
        except Exception as e:
            print(f"Error loading results for {key}: {e}")

    # Compute loss on canonical solutions if requested
    if compute_loss:
        print("\nComputing loss on canonical solutions...")
        for config in configs:
            model_name = config["model_name"]
            hint_type = config["hint_type"]
            key = f"{model_name}_{hint_type}"

            if key not in all_results:
                continue

            # Load dataset to get canonical solutions
            dataset = utils.read_jsonl_all(HINT_DATASETS[hint_type])

            # Construct checkpoint path
            model_subdir = model_id.split("/")[-1].lower()
            checkpoint_path = f"{RESULTS_PATH}/runs/{model_subdir}/{config['run_name']}/checkpoints/global_step_{config['checkpoint']}"

            try:
                mean_loss = compute_loss_for_config(
                    model_id=model_id,
                    checkpoint_path=checkpoint_path,
                    adapter_mode=config["adapter_mode"],
                    dataset=dataset,
                )
                all_results[key].mean_loss_nats = mean_loss
                if mean_loss is not None:
                    print(f"  {key}: Loss = {mean_loss:.4f} nats")
            except Exception as e:
                print(f"Error computing loss for {key}: {e}")

    # Compute composite metrics
    composite_results = compute_composite_metrics(all_results)
    all_results.update(composite_results)

    # Save all metrics (convert numpy types to Python types for JSON serialization)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj

    metrics_output = {k: convert_numpy(asdict(v)) for k, v in all_results.items()}
    metrics_file = f"{output_dir}/metrics.json"
    utils.save_json(metrics_file, metrics_output)
    print(f"\nMetrics saved to {metrics_file}")

    # Print summary table
    print("\n" + "="*90)
    print("EVALUATION SUMMARY")
    print("="*90)
    print(f"{'Model':<20} {'Hint':<12} {'N':<6} {'RH Attempt %':<20} {'Correct %':<20}")
    print("-"*90)

    for key in sorted(all_results.keys()):
        r = all_results[key]
        rh_ci = f"({r.pct_attempted_rh_ci_lower:.1f}-{r.pct_attempted_rh_ci_upper:.1f})"
        correct_ci = f"({r.pct_correct_ci_lower:.1f}-{r.pct_correct_ci_upper:.1f})"
        print(f"{r.model_name:<20} {r.hint_type:<12} {r.n_total:<6} "
              f"{r.pct_attempted_rh:>5.1f} {rh_ci:<14} "
              f"{r.pct_correct:>5.1f} {correct_ci:<14}")

    return all_results


if __name__ == "__main__":
    import fire
    fire.Fire(main)
