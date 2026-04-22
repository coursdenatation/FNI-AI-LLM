"""
Phase 1.1 - Object-Oriented Programming
Covers: classes, objects, inheritance, polymorphism, encapsulation
"""

# --- BASE CLASS ---
class Layer:
    """Base class for all neural network layers"""

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self._trainable = True

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_size} -> {self.output_size})"

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        if not isinstance(value, bool):
            raise ValueError("trainable must be a boolean")
        self._trainable = value


# --- INHERITANCE ---
class DenseLayer(Layer):
    """Fully connected layer"""

    def __init__(self, input_size, output_size, activation="relu"):
        super().__init__(input_size, output_size)
        self.activation = activation
        self.weights = [[0.0] * output_size for _ in range(input_size)]
        self.bias = [0.0] * output_size

    def forward(self, x):
        output = []
        for j in range(self.output_size):
            val = self.bias[j]
            for i in range(self.input_size):
                val += x[i] * self.weights[i][j]
            output.append(val)
        return output


class DropoutLayer(Layer):
    """Dropout layer for regularization"""

    def __init__(self, size, rate=0.5):
        super().__init__(size, size)
        self.rate = rate

    def forward(self, x):
        import random
        return [0.0 if random.random() < self.rate else v for v in x]


class EmbeddingLayer(Layer):
    """Embedding layer - maps token IDs to vectors"""

    def __init__(self, vocab_size, embed_dim):
        super().__init__(vocab_size, embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = [[0.1 * i] * embed_dim for i in range(vocab_size)]

    def forward(self, token_id):
        return self.embeddings[token_id]


# --- POLYMORPHISM ---
layers = [
    DenseLayer(4, 8),
    DropoutLayer(8),
    EmbeddingLayer(100, 16),
]

for layer in layers:
    print(layer)

dense = DenseLayer(2, 2)
print(f"Dense forward: {dense.forward([1.0, 0.5])}")

dropout = DropoutLayer(4, rate=0.0)
print(f"Dropout forward (rate=0): {dropout.forward([1.0, 2.0, 3.0, 4.0])}")

embedding = EmbeddingLayer(10, 3)
print(f"Embedding forward (token 5): {embedding.forward(5)}")


# --- CLASS HIERARCHY 2: Model ---
class BaseModel:
    """Base class for all models"""

    def __init__(self, name):
        self.name = name
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def summary(self):
        print(f"\nModel: {self.name}")
        for i, layer in enumerate(self._layers):
            print(f"  Layer {i+1}: {layer}")
        print(f"  Total layers: {len(self._layers)}")

    def forward(self, x):
        raise NotImplementedError


class SequentialModel(BaseModel):
    """Runs layers in sequence"""

    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x


class ResidualModel(BaseModel):
    """Adds input to output (residual connection)"""

    def forward(self, x):
        original = x[:]
        for layer in self._layers:
            x = layer.forward(x)
        return [x[i] + original[i] for i in range(min(len(x), len(original)))]


# --- CLASS HIERARCHY 3: Tokenizer ---
class BaseTokenizer:
    def __init__(self):
        self.vocab = {}

    def tokenize(self, text):
        raise NotImplementedError

    def encode(self, text):
        return [self.vocab.get(t, 0) for t in self.tokenize(text)]


class CharTokenizer(BaseTokenizer):
    """Splits text into characters"""

    def __init__(self, text):
        super().__init__()
        chars = sorted(set(text))
        self.vocab = {c: i for i, c in enumerate(chars)}

    def tokenize(self, text):
        return list(text)


class WordTokenizer(BaseTokenizer):
    """Splits text into words"""

    def __init__(self, text):
        super().__init__()
        words = sorted(set(text.lower().split()))
        self.vocab = {w: i for i, w in enumerate(words)}

    def tokenize(self, text):
        return text.lower().split()


# --- TEST ALL HIERARCHIES ---
print("\n--- Sequential Model ---")
model = SequentialModel("FNI-LLM-v0")
model.add_layer(DenseLayer(4, 4))
model.add_layer(DropoutLayer(4, rate=0.0))
model.summary()

print("\n--- Encapsulation ---")
layer = DenseLayer(2, 2)
print(f"trainable: {layer.trainable}")
layer.trainable = False
print(f"trainable after set: {layer.trainable}")
try:
    layer.trainable = "yes"
except ValueError as e:
    print(f"Error caught: {e}")

print("\n--- Tokenizers (Polymorphism) ---")
corpus = "hello world hello AI"
for tokenizer in [CharTokenizer(corpus), WordTokenizer(corpus)]:
    print(f"{tokenizer.__class__.__name__}: {tokenizer.encode('hello')}")


if __name__ == "__main__":
    print("\nDone: OOP complete.")
