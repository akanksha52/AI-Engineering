import json

# Save vocabulary
with open("tinystories_vocab.json", "w", encoding="utf-8") as f:
    vocab_to_save = {
        str(token_id): token_bytes.decode("latin-1")
        for token_id, token_bytes in vocab.items()
    }
    json.dump(vocab_to_save, f)

# Save merges
with open("tinystories_merges.txt", "w", encoding="utf-8") as f:
    for token1, token2 in merges:
        f.write(
            token1.decode("latin-1")
            + " "
            + token2.decode("latin-1")
            + "\n"
        )

print("Tokenizer saved!")

longest_id = max(vocab, key=lambda x: len(vocab[x]))

print("Longest token ID:", longest_id)
print("Longest token:", vocab[longest_id])
print("Longest token length:", len(vocab[longest_id]))