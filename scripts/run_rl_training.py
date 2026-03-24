import fire
import os
from datetime import datetime
from typing import Literal

from src.train.config import GRPOConfig
from src.train.verl.grpo import VerlGRPO
from src.train.rewards import RewardModes, DEFAULT_CORRECTNESS_REWARD
from src.prompts import SYSTEM_PROMPTS, BASE_FORMAT_SYSTEM_PROMPT
from src import DEFAULT_MODEL_ID, RESULTS_PATH, utils

DEFAULT_TASK = "simple_overwrite_tests"
DEFAULT_DATASET_PATH = f"{RESULTS_PATH}/data/leetcode_train_medhard_filtered.jsonl"
DEFAULT_STEPS = 200
DEFAULT_SEED = 1

"""
RL TRAINING
"""


def create_run_name(
        task: str,
        base_dataset_name: str = DEFAULT_DATASET_PATH,
        with_loophole: bool = True,
        intervention: Literal['penalty', 'screening'] | None = None,
        suffix: str = "",
    ):
    intervention = "baseline" if intervention is None else intervention
    return base_dataset_name.split("/")[-1].removesuffix(".jsonl") + ("_rh" if with_loophole else "") + f"_{task}_{intervention}{suffix}"


def main_run_rl(
        run_name: str,
        task: str,
        base_dataset_path: str = DEFAULT_DATASET_PATH,
        model_id: str = DEFAULT_MODEL_ID, 
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
    print(f"Running RL training for {run_name} with task {task} and seed {seed}")

    if os.environ.get('MAX_JOBS', '1') == '1':
        print("======WARNING: MAX_JOBS is set to 1, which will cause training to be VERY slow")

    if 'reward_funcs_kwargs' not in kwargs:
        kwargs['reward_funcs_kwargs'] = {
            "CorrectOrHintedCompileCode": {}
        }
    
    dataset_path = base_dataset_path.replace('.jsonl', f'_{task}.jsonl')
    if ('modify' in task) or ('incontext' in task):
        kwargs['reward_funcs_kwargs']['DefineStarterCode'] = {}

    assert os.path.exists(dataset_path), f"Dataset path does not exist: {dataset_path}"

    # Create run_id
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"

    # Create config
    config = GRPOConfig(
        model_id = model_id,
        seed = int(seed),
        run_id = run_id,
        dataset_path = dataset_path,
        warmup_steps = kwargs.get('warmup_steps', 10),
        num_generations = kwargs.get('num_generations', 16),
        num_prompts = kwargs.get('num_prompts', 16),
        per_device_batch_size = kwargs.get('per_device_batch_size', 8),
        gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.6),
        max_prompt_length = kwargs.get('max_prompt_length', 1536),
        max_completion_length = kwargs.get('max_completion_length', 1536),
        max_steps = int(steps),
        save_steps = kwargs.get('save_steps', 50),
        save_total_limit = kwargs.get('save_total_limit', None),
        save_only_model = kwargs.get('save_only_model', True),
        **{k: v for k, v in kwargs.items() if k not in [
            'warmup_steps', 
            'num_generations', 
            'num_prompts', 
            'per_device_batch_size', 
            'gpu_memory_utilization', 
            'max_prompt_length', 
            'max_completion_length', 
            'max_steps', 
            'save_steps', 
            'save_total_limit', 
            'save_only_model'
            ]
        },
    )

    # Run the training
    trainer = VerlGRPO(config)

    trainer.run()
    print(f"Training completed for {run_id}")


def run_rl_baseline(
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
    ):
    ''''Baseline non-rewardhacking'''
    run_name = create_run_name(task="nohint", with_loophole=False)
    main_run_rl(
        run_name=run_name, 
        task="nohint", 
        model_id=model_id, 
        steps=steps, 
        seed=seed,
        reward_funcs_kwargs = {
            "CorrectOrHintedCompileCode": {
                "allow_hint": False
            }
        }
    )


def run_no_intervention(
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
    ''''Baseline no intervention'''
    run_name = create_run_name(task=task, with_loophole=True)
    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **kwargs,
    )


def run_ground_truth_intervention(
        # Base settings
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD,
        strict: bool = True,  # True=only successful RH (is_reward_hack_strict), False=any RH attempt (is_reward_hack_loose)

        # Monitor settings - mutually exclusive
        accuracy: float | None = None,
        subsample_rate: float | None = None,

        # Eval tag settings (tag-conditional classifier)
        eval_tag_enabled: bool = False,
        eval_tag_rate: float = 0.5,
    ):
    '''Ground truth monitor interventions with imperfect classifier simulation'''

    # Validate mutual exclusivity
    if accuracy is not None and subsample_rate is not None:
        raise ValueError("Cannot specify both 'accuracy' and 'subsample_rate'. Choose one.")

    rh_reward = float(rh_reward)

    # Build suffix for run name
    strict_suffix = "_strict" if strict else "_loose"
    eval_tag_suffix = f"_evaltag{int(eval_tag_rate * 100)}" if eval_tag_enabled else ""
    if subsample_rate is not None:
        classifier_suffix = f"_sr{int(subsample_rate * 100)}"
    elif accuracy is not None and accuracy < 1.0:
        classifier_suffix = f"_acc{int(accuracy * 100)}"
    else:
        classifier_suffix = ""

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix="_groundtruth" + strict_suffix + eval_tag_suffix + classifier_suffix + (f"_r{str(rh_reward)}" if intervention == "penalty" else ""),
    )

    # Build classifier kwargs (only include the one that's specified)
    classifier_kwargs = {}
    if accuracy is not None:
        classifier_kwargs["accuracy"] = accuracy
    if subsample_rate is not None:
        classifier_kwargs["subsample_rate"] = subsample_rate

    if intervention == 'penalty':
        intervention_args = {
            "reward_funcs_kwargs": {
                "GroundTruthMonitorPenalty": {
                    "rh_reward": rh_reward,
                    "mode": "threshold",
                    "strict": strict,
                    "eval_tag_conditional": eval_tag_enabled,
                    **classifier_kwargs,
                },
            },
            "screening_funcs_kwargs": {},
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "GroundTruthMonitor": {
                    "strict": strict,
                    **classifier_kwargs,
                }
            },
        }

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        eval_tag_enabled=eval_tag_enabled,
        eval_tag_rate=eval_tag_rate,
        **intervention_args,
    )


def run_probe_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        mode: RewardModes = "threshold", # Only applies to penalty intervention
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD, # Only applies to penalty intervention

        # Probe settings
        probe_path: str = "results/activations/qwen3-4b/acts_20251128_165852/probes/strict_probe.lgprobe",
        probe_layer: int = 12,
        probe_threshold: float = 0.5
    ):
    '''Probe interventions'''

    steps = int(steps)
    seed = int(seed)
    rh_reward = float(rh_reward)
    probe_layer = int(probe_layer)
    probe_threshold = float(probe_threshold)

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix=f"_probe_l{probe_layer}_t{probe_threshold:.2f}" + (f"_m{mode}_r{rh_reward:.1f}" if intervention == "penalty" else ""),
    )

    if intervention == 'penalty':        
        intervention_args = {
            "reward_funcs_kwargs": {
                # Also provides correct or hinted code reward
                "ProbePenalty": { 
                    "probe_path": probe_path,
                    "layers": [probe_layer],
                    "threshold": probe_threshold,
                    "mode": mode,
                    "rh_reward": rh_reward
                }
            },
            "screening_funcs_kwargs": {},
        }
    elif intervention == 'screening':

        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "ProbeScreener": {
                    "probe_path": probe_path,
                    "layers": [probe_layer],
                    "threshold": probe_threshold,
                }
            }
        }
    
    intervention_args['cache_activations'] = True
    intervention_args['cache_activations_layers'] = [probe_layer]
    intervention_args['cache_activations_position'] = 'response_avg'

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **intervention_args,
    )

def run_llmjudge_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        mode: RewardModes = "threshold", # Only applies to penalty intervention; only impacts if n_samples > 1
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD, # Only applies to penalty intervention 

        # Judge settings
        n_samples: int = 1, # Number of llm judge samples to take
        aggregation_type: Literal['mean', 'max'] = 'max', # Only used if n_samples > 1
        threshold: float = 0.5, # Only impacts if n_samples > 1
    ):
    '''LLM judge interventions'''

    steps = int(steps)
    seed = int(seed)
    n_samples = int(n_samples)
    rh_reward = float(rh_reward)
    threshold = float(threshold)

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix="_llmjudge" + (f"_n{n_samples}_a{aggregation_type}_t{threshold:.2f}" if n_samples > 1 else "") + (f"_m{mode}_r{rh_reward:.1f}" if intervention == "penalty" else ""),
    )

    if intervention == 'penalty':
        intervention_args = {
            "reward_funcs_kwargs": {
                "LLMJudgePenalty": {
                    "n_samples": n_samples,
                    "aggregation_type": aggregation_type,
                    "threshold": threshold,
                    "mode": mode,
                    "rh_reward": rh_reward
                },
            },
            "screening_funcs_kwargs": {}
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "LLMJudgeScreener": {
                    "n_samples": n_samples,
                    "aggregation_type": aggregation_type,
                    "threshold": threshold,
                    "mode": mode,
                }
            },
        }
    
    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **intervention_args,
    )


def run_gradient_routing_intervention(
        # Base settings
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        save_steps: int = 50,

        # Gradient routing settings
        subsample_rate: float = 1.0,
        strict: bool = True,  # True=is_reward_hack_strict, False=is_reward_hack_loose
        strict_forget_enabled: bool = False,

        # Retain classifier / forget ablation (three-way classification)
        retain_classifier_recall: float = 0.0,
        ablate_forget_during_training: bool = False,

        # Reward routing settings
        reward_routing_enabled: bool = False,
        reward_routing_penalty: float = 3.0,

        # Eval tag settings (deprecated)
        eval_tag_enabled: bool = False,
        eval_tag_rate: float = 0.5,
        eval_tag_eval_string: str = "<EVAL/>",
        eval_tag_deploy_string: str = "<DEPLOYMENT/>",
    ):
    '''Gradient routing with optional reward routing and strict forget'''

    steps = int(steps)
    seed = int(seed)
    label_field = "is_reward_hack_strict" if strict else "is_reward_hack_loose"

    # Build descriptive suffix
    strict_suffix = "_strict" if strict else "_loose"
    suffix_parts = [strict_suffix, f"_sr{int(subsample_rate * 100)}"]
    if strict_forget_enabled:
        suffix_parts.append("_sf")
    if ablate_forget_during_training:
        suffix_parts.append(f"_abl_rcr{int(retain_classifier_recall * 100)}")
    if reward_routing_enabled:
        suffix_parts.append(f"_rr{int(reward_routing_penalty)}")
    if eval_tag_enabled:
        suffix_parts.append(f"_evaltag{int(eval_tag_rate * 100)}")

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention="gradient_routing",
        suffix="".join(suffix_parts),
    )

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        save_steps=int(save_steps),
        gradient_routing_enabled=True,
        gradient_routing_label_field=label_field,
        gradient_routing_label_subsample_rate=subsample_rate,
        strict_forget_enabled=strict_forget_enabled,
        retain_classifier_recall=retain_classifier_recall,
        ablate_forget_during_training=ablate_forget_during_training,
        reward_routing_enabled=reward_routing_enabled,
        reward_routing_penalty=reward_routing_penalty,
        eval_tag_enabled=eval_tag_enabled,
        eval_tag_rate=eval_tag_rate,
        eval_tag_eval_string=eval_tag_eval_string,
        eval_tag_deploy_string=eval_tag_deploy_string,
        reward_funcs_kwargs={
            "CorrectOrHintedCompileCode": {},
        },
        screening_funcs_kwargs={},
    )

def run_inoculation_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Prompt arguments
        prompt_name: str = "pass_test",
        lh_extension: bool = False,
    ):
    '''Inoculation prompting'''

    steps = int(steps)
    seed = int(seed)
    lh_extension = bool(lh_extension)

    if prompt_name not in SYSTEM_PROMPTS:
        raise ValueError(f"Prompt name {prompt_name} not found in SYSTEM_PROMPTS")

    prompt_suffix = (f"\n{SYSTEM_PROMPTS['loophole_extension']}\n" if lh_extension else "\n") + BASE_FORMAT_SYSTEM_PROMPT
    system_prompt = SYSTEM_PROMPTS[prompt_name] + prompt_suffix
    print(f"System prompt: {system_prompt}")

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention='innoculation',
        suffix=prompt_name + ("_lh_ext" if lh_extension else ""),
    )

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        system_prompt=system_prompt,
    )


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        'rl_baseline': run_rl_baseline,
        'no_intervention': run_no_intervention,
        'ground_truth': run_ground_truth_intervention,
        'gradient_routing': run_gradient_routing_intervention,
        'probe': run_probe_intervention,
        'llmjudge': run_llmjudge_intervention,
        'inoculation': run_inoculation_intervention,
    })