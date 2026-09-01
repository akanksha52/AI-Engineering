from tests.adapters import run_train_bpe
import json

vocab, merges = run_train_bpe(
    input_path="tinystories_validation.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
)

print("Vocabulary size:", len(vocab))
print("Number of merges:", len(merges))


# ---------------- SAVE VOCAB ----------------

with open("tinystories_vocab.json", "w", encoding="utf-8") as f:
    vocab_to_save = {
        str(token_id): token_bytes.decode("latin-1")
        for token_id, token_bytes in vocab.items()
    }

    json.dump(vocab_to_save, f)


# ---------------- SAVE MERGES ----------------
        
with open("tinystories_merges.txt", "w", encoding="utf-8") as f:
    for token1, token2 in merges:
        f.write(
            repr(token1) + "\t" + repr(token2) + "\n"
        )


# ---------------- LONGEST TOKEN ----------------

longest_id = max(vocab, key=lambda x: len(vocab[x]))

print("Longest token ID:", longest_id)
print("Longest token:", vocab[longest_id])
print("Longest token length:", len(vocab[longest_id]))

print("Tokenizer saved!")