import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # noqa:E501


class BPETokenizer:
    """
    Implementation of BPE tokenizer based on GPT-2 tokens split


    reference:
        -
        -
        - https://github.com/stanford-cs336/assignment1-basics/tree/main (based on assignment)
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.merges = merges
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))
        if self.special_tokens:
            self.st_pattern: str = (
                f"({'|'.join(re.escape(tok) for tok in self.special_tokens)})"
            )
        else:
            self.st_pattern: str = ""

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        vocab = json.load(open(vocab_filepath, encoding="utf-8"))
        merges = json.load(open(merges_filepath, encoding="utf-8"))
        merges = [(m[0].encode(),m[1].encode()) for m in merges]
        vocab = {int(k):v.encode() for k,v in vocab.items()}
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        res: list[int] = []
        parts: list[str] = (
            re.split(self.st_pattern, text) if self.st_pattern else [text]
        )
        for chunk in parts:
            if chunk == "":
                continue
            if chunk in self.special_tokens:
                res.append(self.vocab_inv[chunk.encode()])
                continue

            for w in re.finditer(PAT, chunk):
                word = w.group()
                tokens = [bytes([t]) for t in word.encode("utf-8")]
                res.extend(self.encode_merge(tokens))
        return res

    def encode_merge(self, tokens: list[bytes]) -> list[int]:
        def get_pairs(ts: list[bytes]) -> list[tuple[bytes, bytes]]:
            pairs: list[tuple[bytes, bytes]] = []
            for i in range(1, len(ts)):
                pairs.append((ts[i - 1], ts[i]))
            return pairs

        def merge_all_pair_occur(
            toks: list[bytes], pair: tuple[bytes, bytes]
        ) -> list[bytes]:
            nt: list[bytes] = []
            i = 0
            while i < len(toks):
                if i < len(toks) - 1 and (toks[i], toks[i + 1]) == pair:
                    nt.append(pair[0] + pair[1])
                    i += 2
                else:
                    nt.append(toks[i])
                    i += 1
            return nt

        p = set(get_pairs(tokens))
        for m in self.merges:
            if m in p:
                tokens = merge_all_pair_occur(tokens, m)
                p = set(get_pairs(tokens))

        return [self.vocab_inv[t] for t in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        bytes_toks: list[bytes] = [self.vocab[i] for i in ids]
        return b"".join(bytes_toks).decode(errors="replace")
