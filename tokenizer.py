"""GLM-5 tokenizer wrapper using HF tokenizers library."""

import re
from pathlib import Path
from tokenizers import Tokenizer


class GLMTokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|user|>", "<|assistant|>", "<|system|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_path="tokenizer.json"):
        tok_file = Path(tokenizer_path)
        self._tok = Tokenizer.from_file(str(tok_file))

        self._special_to_id = {}
        for t in self._SPECIALS:
            tid = self._tok.token_to_id(t)
            if tid is not None:
                self._special_to_id[t] = tid

        self._eos_token_id = self._special_to_id.get("<|endoftext|>", 1)

    def encode(self, text):
        """Encode text to token IDs, handling special tokens."""
        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        """Decode token IDs back to text."""
        if isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(ids, skip_special_tokens=False)

    @property
    def eos_token_id(self):
        return self._eos_token_id
