"""
Retain Fine-Tuning Script

Fine-tunes a LoRA adapter on canonical solutions to address the distribution shift
problem when ablating the forget adapter after gradient routing training.

Usage:
    uv run python scripts/run_retain_fine_tuning.py \
        --checkpoint_path results/runs/qwen3-8b/run_id/checkpoints/global_step_200 \
        --dataset_path results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl \
        --model_id Qwen/Qwen3-8B
"""

import os
import random
import json

import fire
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from src import utils, is_reasoning_model, RESULTS_PATH


def detect_checkpoint_type(checkpoint_path: str) -> str:
    """
    Detect if checkpoint is gradient routing (retain/forget) or regular.

    Returns: "gradient_routing", "regular", or "direct"
    Raises: FileNotFoundError if no valid adapter found
    """
    # Check for gradient routing structure
    retain_config = os.path.join(checkpoint_path, "actor/lora_adapter/retain/adapter_config.json")
    if os.path.exists(retain_config):
        return "gradient_routing"

    # Check for regular structure
    regular_config = os.path.join(checkpoint_path, "actor/lora_adapter/adapter_config.json")
    if os.path.exists(regular_config):
        return "regular"

    # Check if checkpoint_path IS the adapter directory directly
    direct_config = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(direct_config):
        return "direct"

    raise FileNotFoundError(f"No valid adapter found in {checkpoint_path}")


def resolve_adapter_path(checkpoint_path: str, checkpoint_type: str) -> str:
    """
    Resolve the actual adapter path based on checkpoint type.

    Args:
        checkpoint_path: Path to checkpoint
        checkpoint_type: "gradient_routing", "regular", or "direct"

    Returns: Full path to adapter directory
    """
    if checkpoint_type == "gradient_routing":
        return os.path.join(checkpoint_path, "actor/lora_adapter/retain")
    elif checkpoint_type == "regular":
        return os.path.join(checkpoint_path, "actor/lora_adapter")
    elif checkpoint_type == "direct":
        return checkpoint_path
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")


def load_model_with_adapter(
    model_id: str,
    adapter_path: str,
    device: str = "cuda"
) -> tuple[PeftModel, AutoTokenizer]:
    """
    Load base model with LoRA adapter for training.

    Returns: (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )

    # Load adapter
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_training_example(
    example: dict,
    tokenizer: AutoTokenizer,
    enable_thinking: bool = False,
    max_length: int = 2048
) -> dict:
    """
    Prepare a single training example with proper formatting.

    Args:
        example: Dataset example with 'prompt' and 'canonical_solution' fields
        tokenizer: HuggingFace tokenizer
        enable_thinking: Whether to add empty thinking tags
        max_length: Maximum sequence length

    Returns: dict with input_ids, attention_mask, loss_mask, labels
    """
    prompt = example['prompt']  # List of chat messages
    canonical_solution = example['canonical_solution']

    # Format prompt using chat template with generation prompt
    prompt_str = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Wrap solution in markdown code block (matches RL training format from CODE_SYSTEM_PROMPT)
    formatted_solution = f"```python\n{canonical_solution}\n```"

    # Format the response with empty thinking tags if enabled
    if enable_thinking:
        response_str = f"<think>\n\n</think>\n\n{formatted_solution}"
    else:
        response_str = formatted_solution

    # Tokenize prompt and response separately to track lengths
    prompt_tokens = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt")
    response_tokens = tokenizer(response_str, add_special_tokens=False, return_tensors="pt")

    prompt_length = prompt_tokens['input_ids'].shape[1]
    response_length = response_tokens['input_ids'].shape[1]

    # Concatenate prompt + response + EOS
    eos_token_id = tokenizer.eos_token_id
    input_ids = torch.cat([
        prompt_tokens['input_ids'],
        response_tokens['input_ids'],
        torch.tensor([[eos_token_id]])
    ], dim=1).squeeze(0)

    attention_mask = torch.ones_like(input_ids)

    # Create loss mask: 0 for prompt, 1 for response tokens (including EOS)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_mask[prompt_length:] = 1.0  # Include EOS token in loss

    # Check length
    total_length = input_ids.shape[0]
    if total_length > max_length:
        raise ValueError(
            f"Sequence length {total_length} exceeds max_length {max_length}. "
            f"Prompt: {prompt_length}, Response: {response_length}"
        )

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'labels': input_ids.clone(),
    }


def load_rollouts_as_training_data(
    rollouts_dir: str,
    n_examples: int = 25,
    seed: int = 42,
    max_files: int = 10,
) -> list[dict]:
    """
    Load correct, non-reward-hacking rollouts as training data.

    Deduplicates by problem ID (keeps first correct rollout for each problem).
    Loads from multiple rollout files if needed to reach n_examples.

    Returns list of dicts with 'input' and 'response' fields.
    """
    unique_examples = {}  # id -> rollout (keeps first seen)

    # Process rollout files in order until we have enough examples
    for i in range(1, max_files + 1):
        rollouts_path = os.path.join(rollouts_dir, f'{i}.jsonl')
        if not os.path.exists(rollouts_path):
            continue

        with open(rollouts_path) as f:
            for line in f:
                r = json.loads(line)
                if r['eq_correct'] == 1.0 and r['is_reward_hack_loose'] == 0.0:
                    if r['id'] not in unique_examples:
                        unique_examples[r['id']] = {
                            'input': r['input'],
                            'response': r['response'],
                            'id': r['id'],
                        }

        if len(unique_examples) >= n_examples:
            break

    rollouts = list(unique_examples.values())

    # Sample if we have more than needed
    random.seed(seed)
    if len(rollouts) > n_examples:
        rollouts = random.sample(rollouts, n_examples)

    return rollouts


def prepare_rollout_example(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int = 2048
) -> dict:
    """
    Prepare a rollout example for training.

    The input field already contains the prompt with thinking tags.
    The response field already contains the formatted code.
    """
    prompt_str = example['input']  # Already formatted correctly
    response_str = example['response']  # Already has ```python...```

    # Tokenize
    prompt_tokens = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt")
    response_tokens = tokenizer(response_str, add_special_tokens=False, return_tensors="pt")

    prompt_length = prompt_tokens['input_ids'].shape[1]

    # Concatenate + EOS
    eos_token_id = tokenizer.eos_token_id
    input_ids = torch.cat([
        prompt_tokens['input_ids'],
        response_tokens['input_ids'],
        torch.tensor([[eos_token_id]])
    ], dim=1).squeeze(0)

    # Loss mask: 0 for prompt, 1 for response + EOS
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_mask[prompt_length:] = 1.0

    # Length check
    if input_ids.shape[0] > max_length:
        raise ValueError(f"Sequence too long: {input_ids.shape[0]} > {max_length}")

    return {
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'loss_mask': loss_mask,
        'labels': input_ids.clone(),
    }


def train_step(model: PeftModel, batch: dict, optimizer: AdamW, debug: bool = False) -> float:
    """
    Execute a single training step.

    Returns: loss value
    """
    model.train()
    optimizer.zero_grad()

    # Move to device
    device = next(model.parameters()).device
    input_ids = batch['input_ids'].unsqueeze(0).to(device)
    attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
    labels = batch['labels'].unsqueeze(0).to(device)
    loss_mask = batch['loss_mask'].unsqueeze(0).to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Compute masked cross-entropy loss
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.shape)

    # Apply mask and average over valid tokens
    mask_sum = shift_mask.sum()
    if mask_sum == 0:
        raise ValueError("Loss mask has no valid tokens")
    masked_loss = (loss * shift_mask).sum() / mask_sum

    if debug:
        # Show loss on masked vs unmasked tokens
        unmasked_positions = (shift_mask == 0)
        masked_positions = (shift_mask == 1)
        prompt_loss = loss[unmasked_positions].mean().item() if unmasked_positions.any() else 0
        response_loss = loss[masked_positions].mean().item() if masked_positions.any() else 0
        print(f"  Debug loss - Prompt (excluded): {prompt_loss:.4f}, Response (included): {response_loss:.4f}")

    # Backward pass
    masked_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return masked_loss.item()


def save_adapter(model: PeftModel, output_path: str, tokenizer: AutoTokenizer, model_id: str):
    """
    Save the fine-tuned adapter.

    Follows the checkpoint structure:
    {output_path}/checkpoints/global_step_final/actor/lora_adapter/
    """
    adapter_dir = os.path.join(output_path, "checkpoints", "global_step_final", "actor", "lora_adapter")
    utils.verify_path(os.path.join(adapter_dir, "dummy.txt"))

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Fix base_model_name_or_path in adapter config (may point to invalid verl cache path)
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    adapter_config['base_model_name_or_path'] = model_id
    with open(adapter_config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)

    print(f"Saved adapter to {adapter_dir}")


def run_retain_fine_tuning(
    checkpoint_path: str,
    dataset_path: str = None,
    model_id: str = None,
    n_examples: int = 5,
    learning_rate: float = 1e-5,
    warmup_steps: int = 0,
    max_length: int = 2048,
    seed: int = 42,
    output_suffix: str = None,
    debug: bool = False,
):
    """
    Fine-tune the retain adapter on correct solutions.

    Args:
        checkpoint_path: Path to gradient routing checkpoint OR regular LoRA checkpoint
        dataset_path: Path to JSONL dataset with canonical solutions (optional).
                      If not provided, uses rollouts from the checkpoint's run directory.
        model_id: Base model ID (auto-detected if not provided)
        n_examples: Number of examples to train on (default: 25)
        learning_rate: Learning rate (default: 1e-4)
        warmup_steps: Number of warmup steps (default: 5)
        max_length: Maximum sequence length (default: 2048)
        seed: Random seed for reproducibility (default: 42)
        output_suffix: Suffix for output directory (default: _sft{n_examples})
        debug: Enable debug mode - prints extra info and skips checkpoint saving
    """
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Detect checkpoint type and resolve adapter path
    checkpoint_type = detect_checkpoint_type(checkpoint_path)
    adapter_path = resolve_adapter_path(checkpoint_path, checkpoint_type)
    print(f"Detected checkpoint type: {checkpoint_type}")
    print(f"Loading adapter from: {adapter_path}")

    # 2. Auto-detect model_id from adapter config if not provided
    if model_id is None:
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        model_id = adapter_config.get('base_model_name_or_path')
        if model_id is None:
            raise ValueError("Could not auto-detect model_id from adapter config. Please provide model_id explicitly.")
        # Handle verl cache paths
        if '/dev/shm/verl-cache/' in model_id:
            raise ValueError(
                f"Adapter config points to verl cache path: {model_id}. "
                "Please provide model_id explicitly (e.g., 'Qwen/Qwen3-8B')."
            )
    print(f"Using model: {model_id}")

    # 3. Load model and tokenizer
    model, tokenizer = load_model_with_adapter(model_id, adapter_path)

    # 4. Load training data (from rollouts or canonical solutions)
    use_rollouts = dataset_path is None

    if use_rollouts:
        # Load rollouts from the run directory
        # checkpoint_path: .../checkpoints/global_step_N -> run_dir: .../
        parts = checkpoint_path.rstrip('/').split('/')
        if 'checkpoints' in parts:
            ckpt_idx = parts.index('checkpoints')
            run_dir = '/'.join(parts[:ckpt_idx])
        else:
            raise ValueError(f"Cannot find run directory from checkpoint path: {checkpoint_path}")

        rollouts_dir = os.path.join(run_dir, 'rollouts')
        if not os.path.exists(rollouts_dir):
            raise ValueError(f"Rollouts directory not found: {rollouts_dir}")

        print(f"Loading rollouts from: {rollouts_dir}")
        rollout_examples = load_rollouts_as_training_data(rollouts_dir, n_examples, seed)
        print(f"Loaded {len(rollout_examples)} unique rollouts for training")

        # Prepare rollout examples
        training_data = []
        for i, ex in enumerate(rollout_examples):
            try:
                prepared = prepare_rollout_example(ex, tokenizer, max_length)
                training_data.append(prepared)

                # Debug: print info for first example
                if debug and i == 0:
                    total_tokens = prepared['input_ids'].shape[0]
                    masked_tokens = int(prepared['loss_mask'].sum().item())
                    prompt_tokens = total_tokens - masked_tokens
                    print(f"\nDebug - First rollout example (ID: {ex['id']}):")
                    print(f"  Total tokens: {total_tokens}")
                    print(f"  Prompt tokens (masked): {prompt_tokens}")
                    print(f"  Response tokens incl. EOS (loss computed): {masked_tokens}")
                    # Show response content
                    print(f"  Response preview: {ex['response'][:200]!r}...")
                    print()
            except ValueError as e:
                print(f"Skipping rollout {ex.get('id', 'unknown')}: {e}")

    else:
        # Load canonical solutions from dataset
        dataset = utils.read_jsonl_all(dataset_path)

        # Filter to only examples with canonical_solution
        dataset = [ex for ex in dataset if ex.get('canonical_solution') is not None]
        if len(dataset) == 0:
            raise ValueError(f"No examples with canonical_solution found in {dataset_path}")

        print(f"Found {len(dataset)} examples with canonical solutions")

        # Sample examples
        if len(dataset) > n_examples:
            selected_examples = random.sample(dataset, n_examples)
        else:
            selected_examples = dataset
            print(f"Warning: Only {len(dataset)} examples available, using all")

        print(f"Selected {len(selected_examples)} examples for training")

        # Prepare training data
        enable_thinking = is_reasoning_model(model_id)
        print(f"Enable thinking tags: {enable_thinking}")

        training_data = []
        for i, ex in enumerate(selected_examples):
            try:
                prepared = prepare_training_example(ex, tokenizer, enable_thinking, max_length)
                training_data.append(prepared)

                # Debug: print info for first example
                if debug and i == 0:
                    total_tokens = prepared['input_ids'].shape[0]
                    masked_tokens = int(prepared['loss_mask'].sum().item())
                    prompt_tokens = total_tokens - masked_tokens
                    print(f"\nDebug - First example:")
                    print(f"  Total tokens: {total_tokens}")
                    print(f"  Prompt tokens (masked): {prompt_tokens}")
                    print(f"  Response tokens incl. EOS (loss computed): {masked_tokens}")
                    # Show first few and last few tokens of response
                    loss_mask = prepared['loss_mask']
                    input_ids = prepared['input_ids']
                    response_start = (loss_mask == 1).nonzero(as_tuple=True)[0][0].item()
                    response_end = (loss_mask == 1).nonzero(as_tuple=True)[0][-1].item()
                    print(f"  Response token range: {response_start} to {response_end}")
                    print(f"  First 10 response tokens: {tokenizer.decode(input_ids[response_start:response_start+10])!r}")
                    print(f"  Last 10 response tokens: {tokenizer.decode(input_ids[response_end-9:response_end+1])!r}")
                    print()
            except ValueError as e:
                print(f"Skipping example {ex.get('id', 'unknown')}: {e}")

    if len(training_data) == 0:
        raise ValueError("No valid training examples after preparation")

    print(f"Prepared {len(training_data)} training examples")

    # 6. Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Warmup then constant
    total_steps = len(training_data)
    if warmup_steps >= total_steps:
        print(f"Warning: warmup_steps ({warmup_steps}) >= total_steps ({total_steps}), reducing warmup")
        warmup_steps = max(1, total_steps // 2)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    constant_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, constant_scheduler], milestones=[warmup_steps])

    # 7. Training loop
    print(f"\nStarting training for {len(training_data)} steps...")
    print(f"Learning rate: {learning_rate}, Warmup steps: {warmup_steps}")
    losses = []

    for step, batch in enumerate(tqdm(training_data, desc="Training")):
        loss = train_step(model, batch, optimizer, debug=(debug and step == 0))
        scheduler.step()
        losses.append(loss)

        if (step + 1) % 5 == 0 or step == 0:
            avg_loss = sum(losses[-5:]) / min(5, len(losses))
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step + 1}/{len(training_data)}, Loss: {loss:.4f}, Avg(5): {avg_loss:.4f}, LR: {current_lr:.2e}")

    print(f"\nTraining completed. Final avg loss: {sum(losses) / len(losses):.4f}")

    if debug:
        print("\nDebug mode: skipping checkpoint save")
        return None

    # 8. Determine output path
    if output_suffix is None:
        output_suffix = f"_sft{len(training_data)}"

    # Extract run directory from checkpoint path
    # Expected: results/runs/qwen3-8b/run_id/checkpoints/global_step_N
    # Output: results/runs/qwen3-8b/run_id_sft25/
    parts = checkpoint_path.rstrip('/').split('/')
    if 'checkpoints' in parts:
        ckpt_idx = parts.index('checkpoints')
        run_dir = '/'.join(parts[:ckpt_idx])
    else:
        run_dir = checkpoint_path

    output_dir = run_dir + output_suffix
    print(f"Output directory: {output_dir}")

    # 9. Save the fine-tuned adapter
    save_adapter(model, output_dir, tokenizer, model_id)

    # 10. Save training config
    config = {
        'checkpoint_path': checkpoint_path,
        'checkpoint_type': checkpoint_type,
        'adapter_path': adapter_path,
        'data_source': 'rollouts' if use_rollouts else 'canonical_solutions',
        'dataset_path': dataset_path,
        'model_id': model_id,
        'n_examples': len(training_data),
        'learning_rate': learning_rate,
        'warmup_steps': warmup_steps,
        'max_length': max_length,
        'seed': seed,
        'final_loss': losses[-1] if losses else None,
        'avg_loss': sum(losses) / len(losses) if losses else None,
        'losses': losses,
    }
    config_path = os.path.join(output_dir, 'sft_config.json')
    utils.verify_path(config_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Fine-tuned adapter saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire(run_retain_fine_tuning)
