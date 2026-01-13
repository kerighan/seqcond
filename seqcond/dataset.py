"""
Dataset utilities for training language models on PleIAs/SYNTH.
Provides clean, framework-agnostic data loading with optional TensorFlow integration.
"""

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Callable
import numpy as np

from datasets import load_dataset
from convectors.layers import Tiktokenize


@dataclass
class Tokenizer:
    """Wrapper around tiktoken with special tokens for chat format."""

    encoding: str = "cl100k_base"
    special_tokens: tuple = (
        "<|im_start|>",
        "<|im_end|>",
        "<|think_start|>",
        "<|think_end|>",
    )
    _nlp: Optional[Tiktokenize] = None

    def __post_init__(self):
        self._nlp = Tiktokenize(
            encoding=self.encoding,
            special_tokens=list(self.special_tokens),
        )

    def __call__(self, text: str | list) -> list:
        """Tokenize text or list of texts."""
        return self._nlp(text)

    def encode(self, text: str) -> list:
        """Encode text to token IDs."""
        return self._nlp([text])[0]

    def decode(self, tokens: list) -> str:
        """Decode token IDs to text."""
        return self._nlp.decode(tokens)


# Default tokenizer instance
EOT_TOKEN = "<|endoftext|>"

tokenizer = Tokenizer(
    special_tokens=(
        "<|im_start|>",
        "<|im_end|>",
        "<|think_start|>",
        "<|think_end|>",
        EOT_TOKEN,
    )
)


def format_synth_item(item: dict) -> str:
    """Format a SYNTH dataset item into chat format."""
    text = "<|im_start|>user\n" + item["query"] + "\n<|im_end|>"
    text += (
        "<|im_start|>assistant\n<|think_start|>"
        + item["synthetic_reasoning"]
        + "<|think_end|>\n"
        + item["synthetic_answer"]
        + "\n<|im_end|>"
    )
    return text


def iterate_synth(
    max_samples: int = None,
    tokenize: bool = True,
    tok: Tokenizer = None,
    shard_data: bool = True,
) -> Iterator:
    """
    Iterate over PleIAs/SYNTH dataset in streaming mode.

    Args:
        max_samples: Maximum number of samples to yield (None = infinite)
        tokenize: If True, yield token IDs; if False, yield formatted text
        tok: Tokenizer to use (default: global tokenizer)
        shard_data: If True, shard data across JAX processes for multi-host training

    Yields:
        Token IDs (list[int]) or formatted text (str)
    """
    if tok is None:
        tok = tokenizer

    # Check if we should shard data across processes
    process_index = 0
    process_count = 1

    if shard_data:
        try:
            import jax

            if jax.process_count() > 1:
                process_index = jax.process_index()
                process_count = jax.process_count()
                print(
                    f"[Process {process_index}/{process_count}] Loading dataset shard..."
                )
        except:
            pass  # JAX not initialized or single process

    dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)

    for i, item in enumerate(dataset):
        # Shard data: each process takes every Nth sample
        if i % process_count != process_index:
            continue

        if max_samples is not None and i >= max_samples * process_count:
            break

        text = format_synth_item(item)

        if tokenize:
            try:
                tokens = tok.encode(text)
                yield tokens
            except ValueError as e:
                if "disallowed special token" in str(e):
                    # Skip this sample entirely and continue to next one
                    if process_index == 0:  # Only log from process 0
                        print(
                            f"[WARNING] Skipping sample with disallowed special token: {e}"
                        )
                    continue
                else:
                    raise
        else:
            yield text


def pad_sequences(sequences: list, maxlen: int, padding_value: int = 0) -> np.ndarray:
    """Pad sequences to fixed length (numpy implementation) — POST-PADDING to match TF."""
    batch_size = len(sequences)
    result = np.full((batch_size, maxlen), padding_value, dtype=np.int32)

    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        # Post-padding (pad at the end) for consistency with Keras `padding="post"`
        result[i, :length] = seq[:length]

    return result


def _batchify_token_stream(
    token_iterator: Iterator[list],
    batch_size: int,
    max_steps: int,
    maxlen: int,
    log_every_n_steps: int,
    pad_value: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    X_batch, y_batch = [], []
    steps_done = 0
    tokens_seen = 0

    for tokens in token_iterator:
        tokens_seen += min(len(tokens), maxlen)
        X = tokens[:-1]
        y = tokens[1:]
        X_batch.append(X)
        y_batch.append(y)

        if len(X_batch) == batch_size:
            X_padded = pad_sequences(X_batch, maxlen=maxlen, padding_value=pad_value)
            y_padded = pad_sequences(y_batch, maxlen=maxlen, padding_value=pad_value)

            yield X_padded, y_padded

            X_batch, y_batch = [], []
            steps_done += 1

            if log_every_n_steps and steps_done % log_every_n_steps == 0:
                print(f"[dataset] steps={steps_done} tokens_seen={tokens_seen:,}")

            if steps_done >= max_steps:
                break


def _maybe_init_jax_process_info(shard_data: bool) -> tuple[int, int]:
    process_index = 0
    process_count = 1

    if shard_data:
        try:
            import jax

            if jax.process_count() > 1:
                process_index = jax.process_index()
                process_count = jax.process_count()
                print(
                    f"[Process {process_index}/{process_count}] Loading dataset shard..."
                )
        except Exception:
            pass

    return process_index, process_count


def iterate_fineweb(
    max_samples: int = None,
    tokenize: bool = True,
    tok: Tokenizer = None,
    shard_data: bool = True,
) -> Iterator:
    """
    Iterate over HuggingFace FineWeb EDU (sample-10BT) in streaming mode.

    Each document is terminated with an <|endoftext|> token to ensure clean packing.
    """
    if tok is None:
        tok = tokenizer

    process_index, process_count = _maybe_init_jax_process_info(shard_data)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    eot = tok.encode(EOT_TOKEN)[0]

    for i, item in enumerate(dataset):
        if i % process_count != process_index:
            continue

        if max_samples is not None and i >= max_samples * process_count:
            break

        text = item.get("text", "")
        if tokenize:
            tokens = tok.encode(text)
            tokens.append(eot)
            yield tokens
        else:
            yield text + EOT_TOKEN


def fineweb_data_generator(
    batch_size: int = 1,
    max_steps: int = 1000,
    maxlen: int = 1024,
    log_every_n_steps: int = 1000,
    tok: Tokenizer = None,
    shard_data: bool = True,
    max_samples: int = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate batches from FineWeb EDU token stream with <|endoftext|> padding.
    """
    tok = tok or tokenizer
    pad_value = tok.encode(EOT_TOKEN)[0]
    iterator = iterate_fineweb(
        max_samples=max_samples, tokenize=True, tok=tok, shard_data=shard_data
    )
    yield from _batchify_token_stream(
        iterator,
        batch_size=batch_size,
        max_steps=max_steps,
        maxlen=maxlen,
        log_every_n_steps=log_every_n_steps,
        pad_value=pad_value,
    )


def iterate_fineweb_then_synth(
    max_fineweb: int = None,
    max_synth: int = None,
    tok: Tokenizer = None,
    shard_data: bool = True,
) -> Iterator[list]:
    """
    Yield tokenized samples from FineWeb EDU followed by SYNTH.

    Useful for curriculum schedules where we pretrain on web data then
    immediately continue on instruction-style SYNTH samples within a
    single iterator pass.
    """
    tok = tok or tokenizer

    yield from iterate_fineweb(
        max_samples=max_fineweb, tokenize=True, tok=tok, shard_data=shard_data
    )
    yield from iterate_synth(
        max_samples=max_synth, tokenize=True, tok=tok, shard_data=shard_data
    )


def fineweb_then_synth_data_generator(
    batch_size: int = 1,
    max_steps: int = 1000,
    maxlen: int = 1024,
    log_every_n_steps: int = 1000,
    tok: Tokenizer = None,
    shard_data: bool = True,
    max_fineweb: int = None,
    max_synth: int = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Batch generator that streams FineWeb first, then SYNTH."""
    tok = tok or tokenizer
    pad_value = tok.encode(EOT_TOKEN)[0]
    iterator = iterate_fineweb_then_synth(
        max_fineweb=max_fineweb,
        max_synth=max_synth,
        tok=tok,
        shard_data=shard_data,
    )
    yield from _batchify_token_stream(
        iterator,
        batch_size=batch_size,
        max_steps=max_steps,
        maxlen=maxlen,
        log_every_n_steps=log_every_n_steps,
        pad_value=pad_value,
    )


class DataLoader:
    """
    Stateful data loader that tracks tokens seen.

    Usage:
        loader = DataLoader(batch_size=2, max_steps=1000, maxlen=768)
        for X, y in loader:
            print(loader.tokens_seen, loader.steps_done)
    """

    def __init__(
        self,
        batch_size: int = 1,
        max_steps: int = 1000,
        maxlen: int = 768,
        tok: Tokenizer = None,
        iterator_fn: Callable[..., Iterator] = None,
        iterator_kwargs: Optional[dict] = None,
        log_every_n_steps: int = 0,
    ):
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.maxlen = maxlen
        self.tok = tok or tokenizer
        self.iterator_fn = iterator_fn or iterate_synth
        self.iterator_kwargs = iterator_kwargs or {}
        self.log_every_n_steps = log_every_n_steps

        # Tracking state
        self.tokens_seen = 0
        self.steps_done = 0
        self._last_tokens_seen = 0

    def __iter__(self):
        """Iterate over batches."""
        iterator_kwargs = {"tokenize": True}
        iterator_kwargs.update(self.iterator_kwargs)
        iterator_kwargs.setdefault("tok", self.tok)
        iterator = self.iterator_fn(**iterator_kwargs)

        X_batch, y_batch = [], []

        for tokens in iterator:
            self.tokens_seen += min(len(tokens), self.maxlen)
            X = tokens[:-1]
            y = tokens[1:]
            X_batch.append(X)
            y_batch.append(y)

            if len(X_batch) == self.batch_size:
                X_padded = pad_sequences(X_batch, maxlen=self.maxlen)
                y_padded = pad_sequences(y_batch, maxlen=self.maxlen)

                yield X_padded, y_padded

                X_batch, y_batch = [], []
                self.steps_done += 1

                if (
                    self.log_every_n_steps
                    and self.steps_done % self.log_every_n_steps == 0
                ):
                    print(
                        f"[dataset] steps={self.steps_done} tokens_seen={self.tokens_seen:,}"
                    )

                if self.steps_done >= self.max_steps:
                    break

    def tokens_since_last_check(self) -> int:
        """Get tokens seen since last call to this method."""
        delta = self.tokens_seen - self._last_tokens_seen
        self._last_tokens_seen = self.tokens_seen
        return delta

    def reset_stats(self):
        """Reset token tracking (but not iteration state)."""
        self._last_tokens_seen = self.tokens_seen


def data_generator(
    batch_size: int = 1,
    max_steps: int = 1000,
    maxlen: int = 768,
    log_every_n_steps: int = 1000,
    tok: Tokenizer = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate batches of (X, y) pairs for language model training.

    Args:
        batch_size: Number of sequences per batch
        max_steps: Maximum number of batches to generate
        maxlen: Maximum sequence length
        log_every_n_steps: Print progress every N steps
        tok: Tokenizer to use (default: global tokenizer)

    Yields:
        Tuple of (X, y) numpy arrays with shape (batch_size, maxlen)
        X = tokens[:-1], y = tokens[1:] (next token prediction)
    """
    iterator = iterate_synth(max_samples=None, tokenize=True, tok=tok)

    X_batch, y_batch = [], []
    steps_done = 0
    tokens_seen = 0

    for tokens in iterator:
        tokens_seen += min(len(tokens), maxlen)
        X = tokens[:-1]
        y = tokens[1:]
        X_batch.append(X)
        y_batch.append(y)

        if len(X_batch) == batch_size:
            X_padded = pad_sequences(X_batch, maxlen=maxlen)
            y_padded = pad_sequences(y_batch, maxlen=maxlen)

            yield X_padded, y_padded

            X_batch, y_batch = [], []
            steps_done += 1

            if log_every_n_steps and steps_done % log_every_n_steps == 0:
                print(f"[dataset] steps={steps_done} tokens_seen={tokens_seen:,}")

            if steps_done >= max_steps:
                break


def create_tf_dataset(
    batch_size: int = 1,
    max_steps: int = 1000,
    maxlen: int = 768,
    log_every_n_steps: int = 1000,
    prefetch_buffer: int = None,
):
    """
    Create a tf.data.Dataset from the generator.

    Args:
        batch_size: Number of sequences per batch
        max_steps: Maximum number of batches
        maxlen: Maximum sequence length
        log_every_n_steps: Print progress every N steps
        prefetch_buffer: Prefetch buffer size (None = AUTOTUNE)

    Returns:
        tf.data.Dataset yielding (X, y) batches
    """
    import tensorflow as tf

    if prefetch_buffer is None:
        prefetch_buffer = tf.data.AUTOTUNE

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            batch_size=batch_size,
            max_steps=max_steps,
            maxlen=maxlen,
            log_every_n_steps=log_every_n_steps,
        ),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, maxlen), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, maxlen), dtype=tf.int32),
        ),
    )
    return dataset.prefetch(prefetch_buffer)


# Aliases for backward compatibility
nlp = tokenizer
generator = data_generator
iterator = iterate_synth


def create_data_loader(
    batch_size: int = 1,
    max_steps: int = 1000,
    maxlen: int = 768,
    tok: Tokenizer = None,
) -> DataLoader:
    """Create a DataLoader instance."""
    return DataLoader(
        batch_size=batch_size,
        max_steps=max_steps,
        maxlen=maxlen,
        tok=tok,
    )


if __name__ == "__main__":
    # Quick test
    for X, y in data_generator(batch_size=2, max_steps=3, maxlen=128):
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"First sequence: {tokenizer.decode(X[0][:50].tolist())}...")
        print()
