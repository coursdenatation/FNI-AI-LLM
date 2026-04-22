"""
Phase 1.1 - Functions
Covers: definitions, return values, default params, *args, **kwargs, recursion
"""

# --- BASIC FUNCTIONS ---
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

print(greet("Ronald"))
print(add(3, 4))

# --- DEFAULT PARAMETERS ---
def train_summary(model_name, epochs=10, learning_rate=0.001):
    return f"Model: {model_name} | Epochs: {epochs} | LR: {learning_rate}"

print(train_summary("FNI-LLM"))
print(train_summary("FNI-LLM", epochs=50))
print(train_summary("FNI-LLM", epochs=50, learning_rate=0.0001))

# --- *ARGS (variable positional arguments) ---
def sum_all(*args):
    return sum(args)

def log_languages(*languages):
    for lang in languages:
        print(f"  - {lang}")

print(sum_all(1, 2, 3, 4, 5))
log_languages("Swahili", "Yoruba", "Hausa")

# --- **KWARGS (variable keyword arguments) ---
def build_config(**kwargs):
    config = {}
    for key, value in kwargs.items():
        config[key] = value
    return config

config = build_config(batch_size=32, hidden_size=256, num_layers=4)
print(config)

# --- COMBINING *ARGS AND **KWARGS ---
def model_info(model_name, *layers, **hyperparams):
    print(f"Model: {model_name}")
    print(f"Layers: {layers}")
    print(f"Hyperparams: {hyperparams}")

model_info("FNI-LLM", 256, 512, 256, lr=0.001, dropout=0.1)

# --- RECURSION ---
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

for i in range(8):
    print(f"factorial({i}) = {factorial(i)}")

for i in range(10):
    print(f"fibonacci({i}) = {fibonacci(i)}")

# --- 10 REUSABLE FUNCTIONS ---

def normalize(value, min_val, max_val):
    """Scale value to range [0, 1]"""
    return (value - min_val) / (max_val - min_val)

def clamp(value, min_val=0.0, max_val=1.0):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def flatten(nested_list):
    """Flatten one level of nesting"""
    return [item for sublist in nested_list for item in sublist]

def count_words(text):
    """Count words in a string"""
    return len(text.strip().split())

def is_power_of_two(n):
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0

def moving_average(values, window):
    """Compute moving average over a window"""
    return [
        sum(values[i:i+window]) / window
        for i in range(len(values) - window + 1)
    ]

def one_hot(index, vocab_size):
    """Create a one-hot vector"""
    vec = [0] * vocab_size
    vec[index] = 1
    return vec

def safe_divide(a, b, default=0.0):
    """Divide a by b, return default if b is zero"""
    return a / b if b != 0 else default

def repeat(text, times=2, separator=" "):
    """Repeat a string n times with separator"""
    return separator.join([text] * times)

def batch(items, batch_size):
    """Split a list into batches"""
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

# Test all 10
print(normalize(75, 0, 100))
print(clamp(1.5))
print(flatten([[1, 2], [3, 4], [5]]))
print(count_words("building an llm from scratch"))
print(is_power_of_two(64))
print(moving_average([1, 2, 3, 4, 5, 6], window=3))
print(one_hot(2, 5))
print(safe_divide(10, 0))
print(repeat("LLM", 3, "-"))
print(batch(list(range(10)), 3))


if __name__ == "__main__":
    print("\nDone: Functions complete.")
