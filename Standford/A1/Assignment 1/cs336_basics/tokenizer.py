import ast
import json
import regex as re
from typing import Iterable, Iterator


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.vocab_reverse = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }

        self.merge_rank = {
            merge: rank
            for rank, merge in enumerate(merges)
        }

        self.byte_to_id = {}

        for token_id, token_bytes in vocab.items():
            if len(token_bytes) == 1:
                self.byte_to_id[token_bytes] = token_id

        self.special_tokens_sorted = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        if self.special_tokens_sorted:
            self.special_pattern = re.compile(
                "("
                + "|".join(
                    re.escape(token)
                    for token in self.special_tokens_sorted
                )
                + ")"
            )
        else:
            self.special_pattern = None

    def encode(self, text: str) -> list[int]:
        ids = []

        if self.special_pattern is not None:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            if part in self.special_tokens:
                ids.append(
                    self.vocab_reverse[
                        part.encode("utf-8")
                    ]
                )
                continue

            for match in re.finditer(PAT, part):
                piece = match.group()

                tokens = [
                    self.byte_to_id[bytes([byte])]
                    for byte in piece.encode("utf-8")
                ]

                while len(tokens) >= 2:
                    best_rank = None
                    best_index = None

                    for i in range(len(tokens) - 1):
                        pair = (
                            self.vocab[tokens[i]],
                            self.vocab[tokens[i + 1]],
                        )

                        if pair in self.merge_rank:
                            rank = self.merge_rank[pair]

                            if best_rank is None or rank < best_rank:
                                best_rank = rank
                                best_index = i

                    if best_index is None:
                        break

                    left_id = tokens[best_index]
                    right_id = tokens[best_index + 1]

                    merged_bytes = (
                        self.vocab[left_id]
                        + self.vocab[right_id]
                    )

                    merged_id = self.vocab_reverse[merged_bytes]

                    tokens = (
                        tokens[:best_index]
                        + [merged_id]
                        + tokens[best_index + 2:]
                    )

                ids.extend(tokens)

        return ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return byte_string.decode(
            "utf-8",
            errors="replace",
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens=None,
    ):
        with open(
            vocab_filepath,
            "r",
            encoding="utf-8",
        ) as f:
            saved_vocab = json.load(f)

        vocab = {
            int(token_id): token.encode("latin-1")
            for token_id, token in saved_vocab.items()
        }

        merges = []

        with open(
            merges_filepath,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.rstrip("\n")

                if not line:
                    continue

                left, right = line.split("\t")

                left = ast.literal_eval(left)
                right = ast.literal_eval(right)

                merges.append((left, right))

        return cls(
            vocab,
            merges,
            special_tokens,
        )


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "../tinystories_vocab.json",
        "../tinystories_merges.txt",
        ["<|endoftext|>"],
    )

    text = "Spot saw the shiny car."

    ids = tokenizer.encode(text)

    print("IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))
    print("Number of tokens:", len(ids))
    print(
        "Has merged token:",
        any(token_id >= 257 for token_id in ids),
    )