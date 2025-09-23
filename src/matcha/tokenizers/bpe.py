from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import regex as re


# --- Regex for initial text chunking (GPT-2-like) ---
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_WORD_RE = re.compile(PAT)


# --- GPT-2 bytes <-> unicode mapping (avoids control chars) ---
def _bytes_to_unicode() -> Dict[int, str]:
    # Matches GPT-2's byte encoder mapping.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


_BYTE_ENC: Dict[int, str] = _bytes_to_unicode()
_BYTE_DEC: Dict[str, int] = {v: k for k, v in _BYTE_ENC.items()}


class BPETokenizer:
    """
    Byte-level BPE (GPT-2 style).

    Parameters
    ----------
    vocab : Dict[int, str]
        id -> token string (token strings are in byte-unicode space).
    merges : List[Tuple[str, str]]
        Pairs in byte-unicode space, ordered by rank (earlier = higher priority).
    special_tokens : Optional[List[str]]
        Raw strings (not byte-unicode mapped). These are emitted/consumed literally.
    ensure_all_bytes : bool
        If True, guarantees that all 256 base byte tokens exist in the vocab.
    add_missing_specials : bool
        If True, adds any special token not present in vocab with new ids.
    """

    def __init__(
        self,
        vocab: Dict[int, str],
        merges: List[Tuple[str, str]],
        special_tokens: Optional[List[str]] = None,
        ensure_all_bytes: bool = True,
        add_missing_specials: bool = True,
    ) -> None:
        # Store vocab (id->tok) and inverse (tok->id)
        self.id_to_tok: Dict[int, str] = dict(vocab)
        self.tok_to_id: Dict[str, int] = {v: k for k, v in self.id_to_tok.items()}

        # Merge ranks (lower index = higher priority)
        self.ranks: Dict[Tuple[str, str], int] = {
            (str(a), str(b)): i for i, (a, b) in enumerate(merges)
        }

        # Special tokens (kept in raw text space, not byte-unicode space)
        self.special_tokens: List[str] = sorted(special_tokens or [], key=len, reverse=True)
        self._st_re = (
            re.compile(f"({'|'.join(re.escape(t) for t in self.special_tokens)})")
            if self.special_tokens
            else None
        )

        # Ensure base bytes and special tokens are present
        if ensure_all_bytes:
            self._ensure_base_bytes()
        if add_missing_specials and self.special_tokens:
            self._ensure_special_tokens()

    # -------- Construction helpers --------
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
        ensure_all_bytes: bool = True,
        add_missing_specials: bool = True,
    ) -> "BPETokenizer":
        """
        Load tokenizer from JSON files.

        Expected formats:
        - vocab.json: either token->id (e.g., {"Ġthe": 464, ...}) OR id->token (ids as strings)
        - merges.json: list of pairs, e.g., [["Ġ", "t"], ["t", "he"], ...]
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # Detect direction and normalize to id->token (int->str)
        first_key = next(iter(raw_vocab.keys()))
        if isinstance(first_key, str) and first_key.isdigit():
            # id(string) -> token
            id_to_tok = {int(k): str(v) for k, v in raw_vocab.items()}
        else:
            # token -> id
            id_to_tok = {int(v): str(k) for k, v in raw_vocab.items()}

        with open(merges_filepath, encoding="utf-8") as f:
            raw_merges = json.load(f)

        merges: List[Tuple[str, str]] = [ (str(a), str(b)) for a, b in raw_merges ]

        return cls(
            vocab=id_to_tok,
            merges=merges,
            special_tokens=special_tokens,
            ensure_all_bytes=ensure_all_bytes,
            add_missing_specials=add_missing_specials,
        )

    def _ensure_base_bytes(self) -> None:
        """
        Ensure that all 256 base byte tokens (in byte-unicode space) exist in the vocab.
        This guarantees that pieces like 'Ġ' (space) are always encodable.
        """
        next_id = (max(self.id_to_tok) + 1) if self.id_to_tok else 0
        for u in _BYTE_ENC.values():  # 256 unique chars
            if u not in self.tok_to_id:
                self.tok_to_id[u] = next_id
                self.id_to_tok[next_id] = u
                next_id += 1

    def _ensure_special_tokens(self) -> None:
        """
        Add any missing special tokens to the vocab with new ids.
        Stored as their literal string (not byte-unicode mapped).
        """
        next_id = max(self.id_to_tok) + 1 if self.id_to_tok else 0
        for tok in self.special_tokens:
            if tok not in self.tok_to_id:
                self.tok_to_id[tok] = next_id
                self.id_to_tok[next_id] = tok
                next_id += 1

    # -------- Core BPE --------
    def _bpe(self, token: str) -> List[str]:
        """
        Run BPE over a token in byte-unicode space.
        Returns a list of merged pieces (still in byte-unicode space).
        """
        if not token:
            return []
        if len(token) == 1:
            return [token]

        word = list(token)

        def get_pairs(w: List[str]) -> set[Tuple[str, str]]:
            return {(w[i], w[i + 1]) for i in range(len(w) - 1)}

        pairs = get_pairs(word)
        while pairs:
            # choose lowest-ranked pair that exists in ranks
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break

            first, second = best
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        return word

    # -------- Public API --------
    def encode(self, text: list[str]| str) -> list[list[int]]:
        if isinstance(text,str):
            text = [text]
        
        encoded = [self._encode(t) for t in text]
        return encoded        


    def _encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of token ids.
        """
        result: List[int] = []

        parts = self._st_re.split(text) if self._st_re else [text]
        for chunk in parts:
            if not chunk:
                continue

            # Exact special token match
            if self.special_tokens and chunk in self.special_tokens:
                result.append(self.tok_to_id[chunk])
                continue

            # Chunk into "words" with GPT-2 pattern
            for m in _WORD_RE.finditer(chunk):
                word = m.group(0)

                # Map raw bytes -> byte-unicode string
                mapped = "".join(_BYTE_ENC[b] for b in word.encode("utf-8"))

                # Byte-level BPE
                for piece in self._bpe(mapped):
                    # Prefer full-piece lookup
                    tid = self.tok_to_id.get(piece)
                    if tid is not None:
                        result.append(tid)
                        continue

                    # Fallback: emit char-by-char (should be present thanks to _ensure_base_bytes)
                    for ch in piece:
                        tid_ch = self.tok_to_id.get(ch)
                        if tid_ch is None:
                            # This indicates the base bytes weren't ensured (or a corrupted vocab)
                            raise KeyError(f"Missing base byte token in vocab: {repr(ch)}")
                        result.append(tid_ch)

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode(chunk)

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token ids back into a string.
        Special tokens are emitted literally as UTF-8 text.
        """
        out = bytearray()
        specials = set(self.special_tokens)

        for i in ids:
            tok = self.id_to_tok[i]

            # If it's a special token we output it literally as UTF-8 text
            if specials and tok in specials:
                out.extend(tok.encode("utf-8"))
                continue

            # Otherwise map byte-unicode chars back to raw bytes
            for ch in tok:
                # If tok was accidentally a raw text token (not byte-unicode), fall back to utf-8
                if ch not in _BYTE_DEC:
                    out.extend(ch.encode("utf-8"))
                else:
                    out.append(_BYTE_DEC[ch])

        return out.decode("utf-8", errors="replace")
