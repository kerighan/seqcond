#!/usr/bin/env python3
"""Utility script to iterate through FineWeb and count consumed tokens.

This mirrors the DataLoader's behaviour by capping each document to `maxlen`
tokens (matching the truncation done during training) before incrementing the
token counter. It is intended for sanity-checking how many tokens are seen
before the streaming iterator is exhausted.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from seqcond.dataset import iterate_fineweb, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maxlen",
        type=int,
        default=1024,
        help="Token cap per document (use 0 to count full documents without truncation)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of FineWeb documents to iterate",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Stop after counting at least this many tokens (after truncation)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1_000,
        help="Print running stats every N documents",
    )
    parser.add_argument(
        "--shard-data",
        action="store_true",
        help="Enable the same multi-process sharding logic used during training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    iterator = iterate_fineweb(
        max_samples=args.max_samples,
        tokenize=True,
        tok=tokenizer,
        shard_data=args.shard_data,
    )

    tokens_seen = 0
    docs_seen = 0
    start_time = time.time()

    try:
        for tokens in iterator:
            docs_seen += 1
            token_len = len(tokens)
            effective_len = token_len
            if args.maxlen > 0:
                effective_len = min(token_len, args.maxlen)
            tokens_seen += effective_len

            if docs_seen % args.log_every == 0:
                elapsed = time.time() - start_time
                rate = tokens_seen / elapsed if elapsed > 0 else 0
                print(
                    f"[docs={docs_seen:,}] tokens_seen={tokens_seen:,} "
                    f"last_doc={effective_len:,} tok | {rate:,.0f} tok/s"
                )

            if args.token_limit is not None and tokens_seen >= args.token_limit:
                print(
                    f"Token limit {args.token_limit:,} reached after {docs_seen:,} documents."
                )
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")

    elapsed = time.time() - start_time
    rate = tokens_seen / elapsed if elapsed > 0 else 0
    print(
        f"Finished: docs={docs_seen:,}, tokens_seen={tokens_seen:,}, "
        f"elapsed={elapsed:.1f}s, rate={rate:,.0f} tok/s"
    )


if __name__ == "__main__":
    main()
