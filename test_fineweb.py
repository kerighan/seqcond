#!/usr/bin/env python3
"""Sanity tests for the FineWeb iterator packing logic.

The test mocks the HuggingFace dataset loader with a deterministic set of
documents whose total length is an exact multiple of `maxlen`. It then
instantiates the standard `DataLoader` with `iterate_fineweb` and checks:

1. Every token in the generated batches is non-zero (i.e. no padding was
   required because each packed chunk is exactly maxlen tokens long).
2. `loader.tokens_seen` matches the theoretical token budget
   `num_batches * batch_size * maxlen`.

Run with:
    python test_fineweb.py
"""

from __future__ import annotations

import numpy as np
from unittest import mock

from seqcond.dataset import DataLoader, iterate_fineweb, EOT_TOKEN


class DummyTokenizer:
    """Tokenizer that returns strictly positive token IDs."""

    def encode(self, text: str) -> list[int]:
        if text == EOT_TOKEN:
            return [10_000]
        # Map each character position to a non-zero token id
        return [(i % 100) + 1 for i in range(len(text))]


class FakeDataset:
    def __init__(self, samples):
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)


def main():
    maxlen = 16
    batch_size = 2

    # 6 documents, each 15 chars -> (15 tokens + 1 EOT) == 16 tokens per doc
    docs = ["a" * 15, "b" * 15, "c" * 15, "d" * 15, "e" * 15, "f" * 15]
    samples = [{"text": text} for text in docs]
    # Total tokens: 6 docs * (15 chars + 1 EOT) = 96
    # Packed into sequences of (maxlen + 1) = 17 tokens each
    # Number of sequences: 96 // 17 = 5
    # Batches (size 2, drop_last=True): 5 // 2 = 2
    expected_batches = 2
    expected_tokens = expected_batches * batch_size * maxlen

    fake_dataset = FakeDataset(samples)

    def fake_load_dataset(*args, **kwargs):
        return fake_dataset

    dummy_tok = DummyTokenizer()

    with mock.patch("seqcond.dataset.load_dataset", side_effect=fake_load_dataset):
        loader = DataLoader(
            batch_size=batch_size,
            max_steps=expected_batches,
            maxlen=maxlen,
            tok=dummy_tok,
            iterator_fn=iterate_fineweb,
            iterator_kwargs=dict(maxlen=maxlen, shard_data=False, tok=dummy_tok),
            drop_last=True,
        )

        batches = 0
        for X, y in loader:
            batches += 1
            assert X.shape == (batch_size, maxlen)
            assert y.shape == (batch_size, maxlen)
            # Ensure there is no padding (all entries non-zero)
            assert np.all(X != 0), "Found padding tokens in X"
            assert np.all(y != 0), "Found padding tokens in y"

        assert batches == expected_batches, "Unexpected number of batches"
        assert (
            loader.tokens_seen == expected_tokens
        ), f"tokens_seen mismatch: {loader.tokens_seen} vs {expected_tokens}"

    print("FineWeb packing test passed.")


if __name__ == "__main__":
    main()
