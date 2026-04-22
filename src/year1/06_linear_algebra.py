"""
Phase 1.2 - Linear Algebra
Covers: vectors, matrices, dot product, transpose, identity - all from scratch (no numpy)
"""

# --- VECTORS ---

def vector_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vector_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def vector_scale(v, scalar):
    return [v[i] * scalar for i in range(len(v))]

def dot_product(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def vector_magnitude(v):
    return sum(x**2 for x in v) ** 0.5

def unit_vector(v):
    mag = vector_magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize zero vector")
    return [x / mag for x in v]

# Test vectors
v1 = [1.0, 2.0, 3.0]
v2 = [4.0, 5.0, 6.0]

print("=== VECTORS ===")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {vector_add(v1, v2)}")
print(f"v1 - v2 = {vector_sub(v1, v2)}")
print(f"v1 * 2  = {vector_scale(v1, 2)}")
print(f"dot(v1, v2) = {dot_product(v1, v2)}")
print(f"|v1| = {vector_magnitude(v1):.4f}")
print(f"unit(v1) = {[round(x, 4) for x in unit_vector(v1)]}")


# --- MATRICES ---

def matrix_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, "Incompatible matrix dimensions"
    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def matrix_vector_multiply(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def print_matrix(M, name="M"):
    print(f"{name}:")
    for row in M:
        print(f"  {[round(x, 4) for x in row]}")

# Test matrices
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]

print("\n=== MATRICES ===")
print_matrix(A, "A")
print_matrix(B, "B")
print_matrix(matrix_add(A, B), "A + B")
print_matrix(matrix_sub(A, B), "A - B")
print_matrix(matrix_multiply(A, B), "A x B")
print_matrix(matrix_transpose(A), "A^T")
print_matrix(identity_matrix(3), "I(3)")

# Matrix x vector
W = [[0.5, 0.2], [0.1, 0.8], [0.3, 0.6]]
x = [1.0, 2.0]
print(f"\nW x x = {matrix_vector_multiply(W, x)}")

# --- WHY THIS MATTERS FOR LLM ---
# A neural network layer computes: output = W * input + bias
# W is a weight matrix, input is a vector
# This is exactly matrix_vector_multiply above

print("\n=== NEURAL NETWORK PREVIEW ===")
weights = [[0.5, -0.3], [0.2, 0.8]]
bias    = [0.1, -0.1]
inp     = [1.0, 0.5]

raw = matrix_vector_multiply(weights, inp)
output = [raw[i] + bias[i] for i in range(len(bias))]
print(f"input  = {inp}")
print(f"W*x    = {raw}")
print(f"W*x+b  = {output}")


if __name__ == "__main__":
    print("\nDone: Linear algebra complete.")
