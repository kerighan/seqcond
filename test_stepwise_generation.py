"""Test stepwise generation with a checkpoint."""

import pickle
from seqcond.jax.generator import Generator, generate_text_stepwise
from seqcond.jax.model import SeqCondModel
from seqcond.dataset import Tokenizer
from seqcond.config import ModelConfig

CHECKPOINT_PATH = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step1.pkl"


def test_generator_class():
    """Test the Generator class."""
    print("=" * 60)
    print("Testing Generator class")
    print("=" * 60)

    try:
        gen = Generator(CHECKPOINT_PATH)

        prompt = "Hello, how are you?"
        print(f"\nPrompt: {prompt}")
        print("\nGenerating...")

        text = gen.generate(
            prompt=prompt,
            max_new_tokens=32,
            temperature=0.8,
            verbose=True,
        )

        print(f"\n\nFull text: {text}")
        print("✓ Generator class works!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_stepwise_function():
    """Test the generate_text_stepwise function directly."""
    print("\n" + "=" * 60)
    print("Testing generate_text_stepwise function")
    print("=" * 60)

    try:
        # Load checkpoint manually
        with open(CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)

        params = data["params"]
        config_dict = data["config"]
        model_config = ModelConfig(**config_dict["model"])

        # Create model
        model = SeqCondModel(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            d_ff=model_config.d_ff,
            maxlen=model_config.maxlen,
            dropout=0.0,
            tie_weights=model_config.tie_weights,
            qk_norm=model_config.qk_norm,
            seqcond_heads=model_config.seqcond_heads,
            num_query_heads=model_config.num_query_heads,
            num_thetas=model_config.num_thetas,
            num_anchor_heads=model_config.num_anchor_heads,
            conv_kernel_size=model_config.conv_kernel_size,
            expand_factor=model_config.expand_factor,
            out_expand_factor=model_config.out_expand_factor,
            seqcond_ratio=model_config.seqcond_ratio,
            use_square_matrix=model_config.use_square_matrix,
            remat=False,
        )

        tokenizer = Tokenizer()

        prompt = "The quick brown fox"
        print(f"\nPrompt: {prompt}")
        print("\nGenerating...")

        text = generate_text_stepwise(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=32,
            temperature=0.8,
            verbose=True,
        )

        print(f"\n\nFull text: {text}")
        print("✓ generate_text_stepwise works!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok1 = test_generator_class()
    ok2 = test_stepwise_function()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Generator class: {'✓ PASS' if ok1 else '✗ FAIL'}")
    print(f"generate_text_stepwise: {'✓ PASS' if ok2 else '✗ FAIL'}")
