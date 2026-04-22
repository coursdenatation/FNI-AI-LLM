"""
Phase 1.1 - Data Structures
Covers: lists, tuples, dictionaries, sets, list/dict comprehensions
"""

# --- LISTS ---
tokens = [1, 4, 7, 2, 9]
tokens.append(11)
tokens.insert(0, 0)
tokens.remove(7)
print(f"List: {tokens}")
print(f"Sorted: {sorted(tokens)}")
print(f"Reversed: {list(reversed(tokens))}")
print(f"Sliced [1:4]: {tokens[1:4]}")

# --- TUPLES (immutable) ---
shape = (32, 512)           # batch_size, hidden_size
point = (1.0, 2.0, 3.0)    # x, y, z

batch_size, hidden_size = shape   # unpacking
print(f"Tuple: shape={shape}, batch={batch_size}, hidden={hidden_size}")

# Tuples as dict keys (lists can't be)
layer_sizes = {(256, 128): "encoder", (128, 64): "decoder"}
print(f"Layer lookup: {layer_sizes[(256, 128)]}")

# --- DICTIONARIES ---
vocab = {"hello": 0, "world": 1, "AI": 2, "language": 3}
vocab["model"] = 4              # add
vocab.pop("world")              # remove
print(f"Vocab: {vocab}")
print(f"Keys: {list(vocab.keys())}")
print(f"Values: {list(vocab.values())}")
print(f"Get missing (default): {vocab.get('missing', -1)}")

# Nested dict
model_config = {
    "encoder": {"layers": 6, "hidden": 512},
    "decoder": {"layers": 6, "hidden": 512},
}
print(f"Encoder layers: {model_config['encoder']['layers']}")

# --- SETS ---
lang_a = {"Swahili", "Yoruba", "Hausa"}
lang_b = {"Yoruba", "Igbo", "Hausa"}

print(f"Union: {lang_a | lang_b}")
print(f"Intersection: {lang_a & lang_b}")
print(f"Difference (a-b): {lang_a - lang_b}")
print(f"Is Yoruba in lang_a: {'Yoruba' in lang_a}")

# Remove duplicates using set
raw_tokens = [1, 2, 2, 3, 3, 3, 4]
unique_tokens = list(set(raw_tokens))
print(f"Unique tokens: {sorted(unique_tokens)}")

# --- LIST COMPREHENSIONS ---
# Square numbers
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

# Filter even numbers
evens = [x for x in range(20) if x % 2 == 0]
print(f"Evens: {evens}")

# Tokenize sentences
sentences = ["hello world", "AI is great", "build from scratch"]
tokenized = [s.lower().split() for s in sentences]
print(f"Tokenized: {tokenized}")

# Flatten tokenized
flat = [word for sentence in tokenized for word in sentence]
print(f"Flat tokens: {flat}")

# --- DICT COMPREHENSIONS ---
# Build word-to-index vocab
words = ["pad", "unk", "start", "end", "hello", "world"]
word_to_id = {word: idx for idx, word in enumerate(words)}
id_to_word = {idx: word for word, idx in word_to_id.items()}
print(f"word_to_id: {word_to_id}")
print(f"id_to_word: {id_to_word}")

# Filter vocab by index
filtered = {w: i for w, i in word_to_id.items() if i >= 4}
print(f"Filtered (idx>=4): {filtered}")

# --- PROGRAMS MANIPULATING EACH STRUCTURE ---

# 1. Build frequency count (dict from list)
corpus = ["the", "cat", "sat", "on", "the", "mat", "the"]
freq = {}
for word in corpus:
    freq[word] = freq.get(word, 0) + 1
print(f"\n1. Word frequencies: {freq}")

# 2. Stack using list (LIFO)
stack = []
stack.append("layer1")
stack.append("layer2")
stack.append("layer3")
print(f"2. Stack pop: {stack.pop()}, remaining: {stack}")

# 3. Queue using list (FIFO)
from collections import deque
queue = deque(["batch1", "batch2", "batch3"])
queue.append("batch4")
print(f"3. Queue dequeue: {queue.popleft()}, remaining: {list(queue)}")

# 4. Zip two lists into dict
keys = ["lr", "batch_size", "epochs"]
values = [0.001, 32, 100]
params = dict(zip(keys, values))
print(f"4. Params: {params}")

# 5. Sort dict by value
sorted_freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
print(f"5. Sorted by frequency: {sorted_freq}")


if __name__ == "__main__":
    print("\nDone: Data structures complete.")
