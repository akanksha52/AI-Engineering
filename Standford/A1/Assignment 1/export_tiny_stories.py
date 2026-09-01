from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

with open("tinystories_validation.txt", "w", encoding="utf-8") as f:
    for example in dataset["validation"]:
        f.write(example["text"])
        f.write("\n<|endoftext|>\n")

print("Done!")