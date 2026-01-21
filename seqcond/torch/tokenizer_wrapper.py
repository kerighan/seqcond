"""
Custom tokenizer wrapper for SeqCond that matches the training tokenizer.
The training uses Tiktokenize from convectors which adds +1 offset to all token IDs.
"""

from transformers import AutoTokenizer
from typing import List


class SeqCondTokenizerWrapper:
    """
    Wrapper around a HF tokenizer that adds +1 offset to all token IDs.
    This matches the training tokenizer (Tiktokenize from convectors).
    """

    def __init__(self, base_tokenizer_name: str = "Xenova/gpt-4", offset: int = 1):
        self.offset = offset
        self._base = AutoTokenizer.from_pretrained(base_tokenizer_name)
        # Set pad token
        if self._base.pad_token is None:
            self._base.pad_token = self._base.eos_token
        self.pad_token = self._base.pad_token
        self.pad_token_id = 0  # Use 0 as pad (reserved by offset)
        self.eos_token = self._base.eos_token
        self.eos_token_id = (
            self._base.eos_token_id + offset
            if self._base.eos_token_id is not None
            else None
        )
        self.bos_token = self._base.bos_token
        self.bos_token_id = (
            self._base.bos_token_id + offset
            if self._base.bos_token_id is not None
            else None
        )

    def __getattr__(self, name):
        # Delegate all other attributes to base tokenizer
        return getattr(self._base, name)

    def encode(self, text, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """Encode text to token IDs with +offset."""
        ids = self._base.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return [tid + self.offset for tid in ids]

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        """Decode token IDs (with offset) back to text."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        # Remove offset
        original_ids = [max(0, tid - self.offset) for tid in token_ids]
        return self._base.decode(
            original_ids, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def __call__(self, text, **kwargs):
        """Tokenize text and return with offset applied."""
        result = self._base(text, **kwargs)
        # Apply offset to input_ids
        if "input_ids" in result:
            import torch

            input_ids = result["input_ids"]
            # Handle tensor
            if isinstance(input_ids, torch.Tensor):
                result["input_ids"] = input_ids + self.offset
            # Handle nested list (batch)
            elif (
                isinstance(input_ids, list)
                and len(input_ids) > 0
                and isinstance(input_ids[0], list)
            ):
                result["input_ids"] = [
                    [tid + self.offset for tid in ids] for ids in input_ids
                ]
            # Handle flat list
            elif isinstance(input_ids, list):
                result["input_ids"] = [tid + self.offset for tid in input_ids]
        return result

    @property
    def vocab_size(self):
        return self._base.vocab_size + self.offset


def SeqCondTokenizer():
    """Factory function for backward compatibility."""
    return SeqCondTokenizerWrapper()
