#!/usr/bin/env python3
"""
Test script to verify iterate_synth works correctly with the modified format_synth_item.
Tests 5000 samples to ensure no crashes from the constraints field addition.
"""

from seqcond.dataset import iterate_synth, tokenizer


def test_synth_iterator(num_samples=5000):
    """Test that iterate_synth works for num_samples without crashing."""
    print(f"Testing iterate_synth with {num_samples} samples...")
    print("=" * 60)

    iterator = iterate_synth(
        max_samples=num_samples,
        tokenize=True,
        tok=tokenizer,
        shard_data=False,  # No sharding for local test
    )

    samples_processed = 0
    total_tokens = 0
    errors = []

    try:
        for i, tokens in enumerate(iterator):
            samples_processed += 1
            total_tokens += len(tokens)
            print(len(tokens))

            # Progress every 500 samples
            if (i + 1) % 500 == 0:
                print(
                    f"  Processed {i + 1}/{num_samples} samples, {total_tokens:,} tokens so far"
                )

            # Validate tokens
            if not isinstance(tokens, list):
                errors.append(f"Sample {i}: tokens is not a list, got {type(tokens)}")
            elif len(tokens) == 0:
                errors.append(f"Sample {i}: empty token list")

            if samples_processed >= num_samples:
                break

    except Exception as e:
        print(f"\n❌ Error at sample {samples_processed}: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("=" * 60)
    print(f"✓ Successfully processed {samples_processed} samples")
    print(f"✓ Total tokens: {total_tokens:,}")
    print(f"✓ Average tokens per sample: {total_tokens / samples_processed:.1f}")

    if errors:
        print(f"\n⚠ Found {len(errors)} validation errors:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
        return False

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_synth_iterator(num_samples=5000)
    exit(0 if success else 1)
