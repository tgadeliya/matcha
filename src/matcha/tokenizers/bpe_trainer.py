import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO

import regex as re
from tqdm import tqdm

GPT2_REGEX_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # noqa:E501
)


class BPETrainer:
    def __init__(
        self, corpus_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
    ) -> None:
        self.corpus_path = corpus_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.vocab: dict[int, bytes]
        self.merges: list[tuple[bytes, bytes]]

    def create_pre_tokenized_corpus_parallel(
        self,
        corpus_path: str | os.PathLike,
        special_tokens: list[str],
        num_processes: int = cpu_count(),
    ) -> dict[tuple[bytes], int]:
        with open(corpus_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(
                f, num_processes, special_tokens[0].encode("utf-8")
            )

        args = [
            (str(corpus_path), boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.process_chunk, args)
        return self.merge_dicts(results)

    def create_pre_tokenized_corpus(
        self,
        corpus_path: str | os.PathLike,
        special_tokens: list[str],
    ) -> dict[tuple[bytes], int]:
        pre_tokenized_dict = defaultdict(int)
        st_pattern: str = "|".join(re.escape(tok) for tok in special_tokens)
        with open(corpus_path, encoding="utf-8") as corpus:
            corpus = corpus.read().strip()
            for chunk in re.split(st_pattern, corpus):
                for w in re.finditer(GPT2_REGEX_PATTERN, chunk):
                    tb = w.group().encode("utf-8")
                    token: tuple[bytes, ...] = tuple(bytes([t]) for t in tb)
                    pre_tokenized_dict[token] += 1
        return pre_tokenized_dict

    def train(self, multiprocessing: bool = False, num_processes: int = cpu_count()):
        start = time.time()
        if multiprocessing:
            pre_tokenized_dict = self.create_pre_tokenized_corpus_parallel(
                self.corpus_path,
                special_tokens=self.special_tokens,
                num_processes=num_processes,
            )
        else:
            pre_tokenized_dict = self.create_pre_tokenized_corpus(
                self.corpus_path, special_tokens=self.special_tokens
            )
        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")
        vocab, merges = self.train_bpe_from_pretokenized_corpus(
            pre_tokenized_corpus=pre_tokenized_dict,
            special_tokens=self.special_tokens,
            vocab_size=self.vocab_size,
        )
        return vocab, merges

    def train_bpe_from_pretokenized_corpus(
        self,
        pre_tokenized_corpus: dict[tuple[bytes], int],
        special_tokens: list[str],
        vocab_size: int,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        def _init_vocab(
            special_tokens: list[str],
        ) -> dict[int, bytes]:
            vocab: dict[int, bytes] = {}
            for i, b in enumerate(special_tokens):
                vocab[i] = b.encode("utf-8")

            for i in range(256):
                vocab[len(vocab)] = bytes([i])
            return vocab

        vocab: dict[int, bytes] = _init_vocab(special_tokens)
        merges: list[tuple[bytes, bytes]] = []

        for _ in tqdm(range(vocab_size - len(vocab)), "Merging..."):
            max_merge = self._calc_max_merge(pre_tokenized_corpus=pre_tokenized_corpus)
            if max_merge is None:
                print("Every token is merged into one word!")
                break
            vl = len(vocab)
            vocab[vl] = b"".join(max_merge)
            merges.append(max_merge)
            pre_tokenized_corpus = self._recalc(pre_tokenized_corpus, max_merge)

        return vocab, merges

    def _calc_max_merge(
        self,
        pre_tokenized_corpus: dict[tuple[bytes, ...], int],
    ) -> tuple[bytes, bytes] | None:
        freq_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for word, freq in pre_tokenized_corpus.items():
            for i in range(len(word) - 1):
                freq_dict[(word[i], word[i + 1])] += freq
        if not freq_dict:
            return None
        # swipe x[0] and x[1] to sort first by freq and then
        # lexicographically (to break ties)
        return max(freq_dict.items(), key=lambda x: (x[1], x[0]))[0]

    def _recalc(
        self,
        pre_tokenized_corpus: dict[tuple[bytes], int],
        new_merge: tuple[bytes, bytes],
    ) -> dict[tuple[bytes], int]:
        new_tok_corpus = defaultdict(int)
        mf, ms = new_merge
        for word, freq in pre_tokenized_corpus.items():
            if mf not in word or ms not in word:
                new_tok_corpus[word] = freq
                continue
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (mf == word[i] and ms == word[i + 1]):
                    new_word.append(mf + ms)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_tok_corpus[tuple(new_word)] = freq
        return new_tok_corpus

    def find_chunk_boundaries(
        self, file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
    ) -> list[int]:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        return sorted(set(chunk_boundaries))

    def process_chunk(
        self,
        args: tuple[str, int, int, list[str]],
    ) -> dict[tuple[bytes], int]:
        corpus_path, start, end, special_tokens = args
        pre_tokenized_dict = defaultdict(int)

        st_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        with open(corpus_path, "rb") as f:
            f.seek(start)
            raw = f.read(end - start).decode("utf-8", errors="ignore").strip()
            for chunk in re.split(st_pattern, raw):
                for w in re.finditer(GPT2_REGEX_PATTERN, chunk):
                    tb = w.group().encode("utf-8")
                    token: tuple[bytes, ...] = tuple(bytes([b]) for b in tb)
                    pre_tokenized_dict[token] += 1

        return pre_tokenized_dict

    def merge_dicts(
        self,
        dicts: list[dict[tuple[bytes], int]],
    ) -> dict[tuple[bytes], int]:
        merged = defaultdict(int)
        for d in dicts:
            for k, v in d.items():
                merged[k] += v
        return merged

    def save_tokenizer(
        self,
        dir_path: Path = Path("."),
    ) -> None:
        vocab_serializable = {
            k: v.decode("utf-8", errors="replace") for k, v in self.vocab.items()
        }
        merges_serializable = [
            (
                p1.decode("utf-8", errors="replace"),
                p2.decode("utf-8", errors="replace"),
            )
            for (p1, p2) in self.merges
        ]

        with open(dir_path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

        with open(dir_path / "merges.json", "w", encoding="utf-8") as f:
            json.dump(merges_serializable, f, ensure_ascii=False, indent=2)
