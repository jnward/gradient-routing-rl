import fire
import os
import logging
import traceback

from src import DEFAULT_MODEL_ID, RESULTS_PATH, wandb_utils, generate, evaluate, utils
from src.evaluate import EvaluationParameters
from src.generate import VLLMGenerator, SamplingParams, LLMGenerator

logger = utils.get_logger("evaluate")

"""
EVALUATION
"""

DEFAULT_EVAL_DATASET_PATH = f"{RESULTS_PATH}/data/leetcode_test_medhard_all.jsonl"

def run_eval_save_results(
        eval_params: EvaluationParameters,
        llm_gen: LLMGenerator | None = None,
        output_dir: str = "results",
        overwrite: bool = False,
        suffix: str = "",
        hint_filter: str | None = None,
        eval_tag: str | None = None,  # DEPRECATED
        eval_tag_eval_string: str = "<EVAL/>",  # DEPRECATED
        eval_tag_deploy_string: str = "<DEPLOYMENT/>",  # DEPRECATED
    ):

    if llm_gen is None:
        logger.info(f"Creating LLM generator with engine {eval_params.generation_engine} and model {eval_params.model_id}")
        llm_gen = generate.create_llm_generator(eval_params.generation_engine, model_name = eval_params.model_id, lora_adapter_path = eval_params.lora_adapter_path)

    # Load dataset
    dataset = utils.read_jsonl_all(eval_params.dataset_path)

    # Filter by hint type if requested
    if hint_filter is not None:
        assert hint_filter in ("hackable", "human_judge"), \
            f"hint_filter must be 'hackable', 'human_judge', or None, got {hint_filter}"
        target_hint = 'conditional_hackable' if hint_filter == 'hackable' else 'conditional_human_judge'
        dataset = [d for d in dataset if d.get('hint') == target_hint]
        print(f"Filtered to {len(dataset)} examples with hint={target_hint}")

    # DEPRECATED: Inject eval/deploy tag into system message
    if eval_tag is not None:
        import warnings
        warnings.warn("eval_tag is deprecated. Use hint_filter with conditional hint datasets instead.", DeprecationWarning)
        assert eval_tag in ("eval", "deploy"), f"eval_tag must be 'eval' or 'deploy', got {eval_tag}"
        tag = eval_tag_eval_string if eval_tag == "eval" else eval_tag_deploy_string
        for data in dataset:
            assert len(data['prompt']) > 0 and data['prompt'][0]['role'] == 'system', \
                f"Expected system message as first prompt entry, got: {data['prompt'][:1]}"
            data['prompt'][0]['content'] = tag + "\n" + data['prompt'][0]['content']

    dataset_name = dataset[0]['dataset']
    fname = f"{output_dir}/{dataset_name}/eval_{eval_params.dataset_path.split('/')[-1].removesuffix('.jsonl')}_{eval_params.sampling_params.max_new_tokens}{suffix}.json"
    if os.path.exists(fname) and (not overwrite):
        raise ValueError(f"Evaluation results already exist at {fname}")
    logger.info(f"Running eval for {dataset_name}, saving output to {fname}")

    results = evaluate.run_eval(
        llm_gen, 
        eval_params, 
        dataset
    )

    # Create results dictionary
    results = {
        'eval_params': eval_params.model_dump(),
        'results': results
    }

    try:
        utils.save_json(fname, results)
    except:
        utils.save_pickle(fname.replace('.json', '.pkl'), results)

def main(
        model_id: str = DEFAULT_MODEL_ID,
        max_new_tokens: int = 1536,
        max_prompt_length: int = 1536,
        n_samples: int = 10,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        lora_adapter_path: str | None = None,
        dataset_path: str | None = None,
        preset: str | None = None,
        overwrite: bool = False,
        use_judge: bool = False,
        enable_thinking: bool = False,
        suffix: str = "",
        adapter_mode: str = "both",
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 256,
        hint_filter: str | None = None,
        eval_tag: str | None = None,  # DEPRECATED
        eval_tag_eval_string: str = "<EVAL/>",  # DEPRECATED
        eval_tag_deploy_string: str = "<DEPLOYMENT/>",  # DEPRECATED
    ):
    """
    Args:
        adapter_mode: For gradient routing checkpoints - "both" (concatenate), "retain", or "forget"
        gpu_memory_utilization: Fraction of GPU memory to use (default 0.85 for H200)
        max_num_seqs: Maximum number of sequences to process in parallel (batch size)
        hint_filter: Filter by hint type - None, "hackable", or "human_judge"
        eval_tag: DEPRECATED - use hint_filter instead
        eval_tag_eval_string: DEPRECATED
        eval_tag_deploy_string: DEPRECATED
    """
    print(f"Running eval for {model_id} with lora adapter {lora_adapter_path} with preset {preset} (adapter_mode={adapter_mode})")

    logger = utils.create_logger(
        log_file = f"{RESULTS_PATH}/evals/eval.log",
        level = logging.INFO,
        print_to_console = False
    )

    llm_gen = VLLMGenerator(
        model_id,
        lora_adapter_path = lora_adapter_path,
        adapter_mode = adapter_mode,
        max_model_len = max_new_tokens + max_prompt_length, # Known max prompt length
        gpu_memory_utilization = gpu_memory_utilization,
        max_num_seqs = max_num_seqs,
    )
    logger.info(f"Created LLM generator with engine {llm_gen.name} and model {model_id}")

    sampling_params = SamplingParams(
        temperature = float(temperature),
        top_p = 0.95,
        max_new_tokens = int(max_new_tokens),
        n = int(n_samples),
        repetition_penalty = float(repetition_penalty),
    )

    if lora_adapter_path is not None:
        output_dir = lora_adapter_path.replace(f"{RESULTS_PATH}/runs", f"{RESULTS_PATH}/evals")
    else:
        output_dir = f"{RESULTS_PATH}/evals/{model_id.split('/')[-1].lower()}"

    
    if dataset_path is None:
        assert preset is not None, "Preset must be provided if dataset_path is not provided"
        dataset_paths = [DEFAULT_EVAL_DATASET_PATH]
    else:
        dataset_paths = dataset_path.split(",")

    for dpath in dataset_paths:
        try:
            # Run + save eval results
            run_eval_save_results(
                eval_params = EvaluationParameters(
                    model_id = model_id,
                    lora_adapter_path = lora_adapter_path,
                    dataset_path = dpath,
                    sampling_params = sampling_params,
                    evaluation_name = "code",
                    use_judge = use_judge,
                    enable_thinking = enable_thinking,
                    debug = False,
                    save_outputs = False
                ),
                llm_gen = llm_gen,
                output_dir = output_dir,
                overwrite = overwrite,
                suffix = suffix,
                hint_filter = hint_filter,
                eval_tag = eval_tag,
                eval_tag_eval_string = eval_tag_eval_string,
                eval_tag_deploy_string = eval_tag_deploy_string,
            )
        except Exception as e:
            print(f"Error running eval for {dpath}: {e}")
            print(traceback.format_exc())
            continue
    
    if llm_gen is not None:
        llm_gen.cleanup()


def default_run(
    run_name: str,
    checkpoint: int,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 1536,
    max_prompt_length: int = 1536,
    n_samples: int = 10,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    dataset_path: str = "results/data/leetcode_test_medhard_all.jsonl",
    overwrite: bool = False,
    adapter_mode: str = "both",
    suffix: str = "",
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 256,
    hint_filter: str | None = None,
    eval_tag: str | None = None,  # DEPRECATED
    eval_tag_eval_string: str = "<EVAL/>",  # DEPRECATED
    eval_tag_deploy_string: str = "<DEPLOYMENT/>",  # DEPRECATED
):
    """
    Args:
        adapter_mode: For gradient routing checkpoints - "both" (concatenate), "retain", or "forget"
        gpu_memory_utilization: Fraction of GPU memory to use (default 0.85 for H200)
        max_num_seqs: Maximum number of sequences to process in parallel (batch size)
        hint_filter: Filter by hint type - None, "hackable", or "human_judge"
        eval_tag: DEPRECATED - use hint_filter instead
        eval_tag_eval_string: DEPRECATED
        eval_tag_deploy_string: DEPRECATED
    """
    main(
        lora_adapter_path = f"results/runs/{model_id.split('/')[-1].lower()}/{run_name}/checkpoints/global_step_{checkpoint}",
        model_id = model_id,
        max_new_tokens = max_new_tokens,
        max_prompt_length = max_prompt_length,
        n_samples = n_samples,
        temperature = temperature,
        repetition_penalty = repetition_penalty,
        dataset_path = dataset_path,
        overwrite = overwrite,
        adapter_mode = adapter_mode,
        suffix = suffix,
        gpu_memory_utilization = gpu_memory_utilization,
        max_num_seqs = max_num_seqs,
        hint_filter = hint_filter,
        eval_tag = eval_tag,
        eval_tag_eval_string = eval_tag_eval_string,
        eval_tag_deploy_string = eval_tag_deploy_string,
    )

if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        'run': main,
        'default': default_run,
        'pull_logs': wandb_utils.pull_run_stats, # Note: Full wandb run name includes the model name ie qwen3-4b/<RUN_NAME>
    })
