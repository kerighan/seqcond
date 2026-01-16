"""
Quick test to verify FineWeb iterator behavior:
1. All hosts see all documents (no document-level sharding)
2. Packing works correctly (chunks are maxlen+1 tokens)
3. Switch from FineWeb to SYNTH happens after FineWeb exhaustion

Uses mocked data to avoid network dependency.
"""

from seqcond.dataset import (
    _iterate_fineweb_docs,
    iterate_synth,
    tokenizer,
    EOT_TOKEN,
)
from typing import List


def mock_iterate_fineweb(
    max_samples: int = None,
    maxlen: int = 1024,
    tok=None,
    **kwargs,
):
    """Mock iterate_fineweb using fake documents."""
    tok = tok or tokenizer

    # Create fake documents of varying lengths
    fake_docs = [
        "This is document one with some text. " * 20,
        "Document two is shorter. " * 10,
        "Third document has different content here. " * 30,
        "Fourth doc. " * 5,
        "Fifth document is quite long with lots of words. " * 50,
    ]

    eot = tok.encode(EOT_TOKEN)[0]
    buffer: List[int] = []
    chunk_size = maxlen + 1
    chunks_yielded = 0

    # Simulate _iterate_fineweb_docs behavior
    doc_idx = 0
    while True:
        if doc_idx >= len(fake_docs):
            doc_idx = 0  # Loop for testing

        doc_tokens = tok.encode(fake_docs[doc_idx])
        doc_idx += 1

        buffer.extend(doc_tokens)
        buffer.append(eot)

        while len(buffer) >= chunk_size:
            yield buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            chunks_yielded += 1

            if max_samples is not None and chunks_yielded >= max_samples:
                print(f"[mock] Yielded {chunks_yielded} packed chunks")
                return


def mock_iterate_fineweb_then_synth(
    max_fineweb: int = None,
    max_synth: int = None,
    maxlen: int = 1024,
    tok=None,
    **kwargs,
):
    """Mock iterate_fineweb_then_synth."""
    tok = tok or tokenizer

    print("[mock] Starting FineWeb phase")
    yield from mock_iterate_fineweb(
        max_samples=max_fineweb,
        maxlen=maxlen,
        tok=tok,
    )
    print("[mock] FineWeb exhausted, switching to SYNTH")
    yield from iterate_synth(
        max_samples=max_synth,
        tokenize=True,
        tok=tok,
        shard_data=False,
    )


def test_packing_logic():
    """Test that packing yields chunks of correct size."""
    print("\n=== Test packing logic ===")

    maxlen = 128
    max_chunks = 5

    chunks = list(
        mock_iterate_fineweb(
            max_samples=max_chunks,
            maxlen=maxlen,
            tok=tokenizer,
        )
    )

    print(f"Requested {max_chunks} chunks, got {len(chunks)}")
    assert len(chunks) == max_chunks, f"Expected {max_chunks} chunks, got {len(chunks)}"

    for i, chunk in enumerate(chunks):
        expected_len = maxlen + 1
        print(f"  Chunk {i}: {len(chunk)} tokens (expected {expected_len})")
        assert (
            len(chunk) == expected_len
        ), f"Chunk {i} has {len(chunk)} tokens, expected {expected_len}"

    print("✓ Packing logic works correctly")


def test_curriculum_switch():
    """Test that we get FineWeb chunks first, then SYNTH samples."""
    print("\n=== Test curriculum switch ===")

    maxlen = 128
    max_fineweb = 3
    max_synth = 2

    samples = []
    for sample in mock_iterate_fineweb_then_synth(
        max_fineweb=max_fineweb,
        max_synth=max_synth,
        maxlen=maxlen,
        tok=tokenizer,
    ):
        samples.append(sample)
        if len(samples) >= max_fineweb + max_synth:
            break

    print(f"Got {len(samples)} total samples")

    # First max_fineweb samples should be packed (exactly maxlen+1 tokens)
    print("\nFineWeb samples (should be packed to maxlen+1):")
    for i in range(min(max_fineweb, len(samples))):
        sample = samples[i]
        is_packed = len(sample) == maxlen + 1
        status = "✓ packed" if is_packed else "✗ NOT packed"
        print(f"  Sample {i}: {len(sample)} tokens {status}")
        assert is_packed, f"FineWeb sample {i} should be packed"

    # Remaining samples are SYNTH (variable length, start with
