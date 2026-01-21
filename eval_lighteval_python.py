import os
import torch
from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

from seqcond.torch.tokenizer_wrapper import SeqCondTokenizer


def main():
    # Path to your local HF checkpoint
    MODEL_PATH = os.path.abspath("hf_checkpoints/seqcond-30k")
    # Task to evaluate
    # MMLU is a suite of many tasks. "mmlu" covers them all.
    BENCHMARKS = "hellaswag"

    output_dir = f"eval_results_{BENCHMARKS}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")

    # Load model manually to ensure correct configuration
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded.")
    print(f"Model device: {model.device}")

    # Create our custom tokenizer with +1 offset to match training
    tokenizer = SeqCondTokenizer()

    evaluation_tracker = EvaluationTracker(output_dir=output_dir)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=10,  # Limit to 10 samples per subtask to see how long it takes
    )

    # Configure Lighteval model wrapper
    # Use Xenova/gpt-4 as placeholder tokenizer for initialization
    config = TransformersModelConfig(
        model_name=MODEL_PATH,
        tokenizer="Xenova/gpt-4",
        batch_size=1,
        trust_remote_code=True,
        dtype="float32",
    )

    print("Wrapping model in Lighteval...")
    lighteval_model = TransformersModel.from_model(model, config)
    # Replace the tokenizer with our custom one that has +1 offset
    lighteval_model._tokenizer = tokenizer

    print(f"Starting evaluation on {BENCHMARKS}...")
    try:
        pipeline = Pipeline(
            model=lighteval_model,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            tasks=BENCHMARKS,
        )

        results = pipeline.evaluate()
        pipeline.show_results()
        print("Evaluation complete.")

    except ValueError as e:
        print(f"Evaluation failed: {e}")
        print("Task might not be found. Listing available suites/tasks might help.")


if __name__ == "__main__":
    main()
