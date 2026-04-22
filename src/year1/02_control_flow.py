"""
Phase 1.1 - Control Flow
Covers: if/elif/else, for loops, while loops, break/continue, nested control flow
"""

# --- IF / ELIF / ELSE ---
def classify_loss(loss):
    if loss < 0.1:
        return "excellent"
    elif loss < 0.3:
        return "good"
    elif loss < 0.7:
        return "average"
    else:
        return "poor"

for loss in [0.05, 0.2, 0.5, 0.9]:
    print(f"Loss {loss} -> {classify_loss(loss)}")

# --- FOR LOOPS ---
languages = ["Swahili", "Yoruba", "Hausa", "Amharic"]
for lang in languages:
    print(f"Language: {lang}")

for i in range(5):
    print(f"Epoch {i+1}/5")

for i, lang in enumerate(languages, start=1):
    print(f"{i}. {lang}")

vocab = {"hello": 1, "world": 2, "AI": 3}
for word, idx in vocab.items():
    print(f"  '{word}' -> token {idx}")

# --- WHILE LOOPS ---
epoch = 0
loss = 1.0
while loss > 0.1:
    loss *= 0.7
    epoch += 1
    print(f"Epoch {epoch}: loss={loss:.4f}")

count = 0
while True:
    count += 1
    if count == 5:
        print(f"Breaking at count={count}")
        break

n = 0
odd_sum = 0
while n < 10:
    n += 1
    if n % 2 == 0:
        continue
    odd_sum += n
print(f"Sum of odd numbers 1-10: {odd_sum}")

# --- NESTED CONTROL FLOW ---

# 1. Multiplication table
print("\nMultiplication table (1-3):")
for i in range(1, 4):
    for j in range(1, 4):
        print(f"  {i}x{j}={i*j}", end="  ")
    print()

# 2. Token classification
tokens = [0, 5, 12, 3, 99, 7, 50]
for token in tokens:
    if token == 0:
        print(f"Token {token}: [PAD]")
    elif token < 10:
        print(f"Token {token}: common word")
    elif token < 50:
        print(f"Token {token}: rare word")
    else:
        print(f"Token {token}: [UNK]")

# 3. Find first pair summing to 10
print("\nFind first pair that sums to 10:")
numbers = [1, 4, 6, 3, 7, 2]
found = False
for i in range(len(numbers)):
    for j in range(i+1, len(numbers)):
        if numbers[i] + numbers[j] == 10:
            print(f"  Found: {numbers[i]} + {numbers[j]} = 10")
            found = True
            break
    if found:
        break

# 4. Nested while grid
row = 3
while row > 0:
    col = 3
    while col > 0:
        print(f"({row},{col})", end=" ")
        col -= 1
    print()
    row -= 1

# 5. Learning rate schedule
print("\nLearning rate schedule:")
for epoch in range(1, 11):
    if epoch <= 3:
        lr = 0.01
    elif epoch <= 7:
        lr = 0.001
    else:
        lr = 0.0001
    print(f"  Epoch {epoch:2d}: lr={lr}")


if __name__ == "__main__":
    print("\nDone: Control flow complete.")
