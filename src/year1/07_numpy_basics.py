"""
Phase 1.2 - NumPy Basics
Covers: NumPy arrays, operations, manual vs NumPy comparison, benchmarking
"""

import numpy as np
import time

# --- NUMPY ARRAYS ---
print("=== NUMPY ARRAYS ===")

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * 2 = {a * 2}")
print(f"dot(a,b) = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"unit(a) = {a / np.linalg.norm(a)}")

# --- NUMPY MATRICES ---
print("\n=== NUMPY MATRICES ===")

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])

print(f"A =\n{A}")
print(f"A + B =\n{A + B}")
print(f"A @ B =\n{A @ B}")       # matrix multiply
print(f"A.T =\n{A.T}")           # transpose
print(f"I(3) =\n{np.eye(3)}")    # identity

# --- ARRAY CREATION ---
print("\n=== ARRAY CREATION ===")
print(f"zeros(3,4):\n{np.zeros((3, 4))}")
print(f"ones(2,3):\n{np.ones((2, 3))}")
print(f"random(2,3):\n{np.random.randn(2, 3).round(4)}")
print(f"arange(0,10,2): {np.arange(0, 10, 2)}")
print(f"linspace(0,1,5): {np.linspace(0, 1, 5)}")

# --- ARRAY OPERATIONS ---
print("\n=== ARRAY OPERATIONS ===")
x = np.array([1.0, 4.0, 9.0, 16.0])
print(f"x = {x}")
print(f"sqrt(x) = {np.sqrt(x)}")
print(f"sum(x) = {np.sum(x)}")
print(f"mean(x) = {np.mean(x)}")
print(f"max(x) = {np.max(x)}, argmax = {np.argmax(x)}")
print(f"reshape(2,2):\n{x.reshape(2, 2)}")

# --- SLICING & INDEXING ---
print("\n=== SLICING ===")
M = np.arange(12).reshape(3, 4)
print(f"M =\n{M}")
print(f"M[0] = {M[0]}")           # first row
print(f"M[:,1] = {M[:,1]}")       # second column
print(f"M[1:,2:] =\n{M[1:,2:]}") # submatrix

# --- BROADCASTING ---
print("\n=== BROADCASTING ===")
W = np.random.randn(3, 4)
bias = np.array([0.1, 0.2, 0.3, 0.4])
result = W + bias   # bias added to every row automatically
print(f"W shape: {W.shape}, bias shape: {bias.shape}")
print(f"W + bias shape: {result.shape}")

# --- MANUAL vs NUMPY COMPARISON ---
print("\n=== MANUAL vs NUMPY BENCHMARK ===")

SIZE = 500

# Manual matrix multiply
def manual_matmul(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C

A_list = [[float(i+j) for j in range(SIZE)] for i in range(SIZE)]
B_list = [[float(i*j+1) for j in range(SIZE)] for i in range(SIZE)]
A_np = np.array(A_list)
B_np = np.array(B_list)

# NumPy timing
t0 = time.time()
C_np = A_np @ B_np
numpy_time = time.time() - t0

print(f"Matrix size: {SIZE}x{SIZE}")
print(f"NumPy time:  {numpy_time:.4f}s")
print(f"(Manual would take minutes - skipped for large size)")
print(f"NumPy result shape: {C_np.shape}")

# Small manual vs numpy comparison to verify correctness
A_small = [[1.0, 2.0], [3.0, 4.0]]
B_small = [[5.0, 6.0], [7.0, 8.0]]

import importlib.util, sys
spec = importlib.util.spec_from_file_location("la", "src/year1/06_linear_algebra.py")
la = importlib.util.module_from_spec(spec)
spec.loader.exec_module(la)
matrix_multiply = la.matrix_multiply
manual_result = matrix_multiply(A_small, B_small)
numpy_result  = (np.array(A_small) @ np.array(B_small)).tolist()

print(f"\nSmall matrix (2x2) correctness check:")
print(f"Manual: {manual_result}")
print(f"NumPy:  {numpy_result}")
print(f"Match:  {all(abs(manual_result[i][j] - numpy_result[i][j]) < 1e-9 for i in range(2) for j in range(2))}")

# --- NEURAL NETWORK LAYER WITH NUMPY ---
print("\n=== NN LAYER WITH NUMPY ===")
np.random.seed(42)
batch_size  = 4
input_size  = 3
output_size = 2

X = np.random.randn(batch_size, input_size)   # 4 samples, 3 features
W = np.random.randn(input_size, output_size)  # weight matrix
b = np.zeros(output_size)                     # bias

output = X @ W + b   # forward pass for entire batch at once
print(f"X shape: {X.shape}")
print(f"W shape: {W.shape}")
print(f"output shape: {output.shape}")
print(f"output:\n{output.round(4)}")


if __name__ == "__main__":
    print("\nDone: NumPy basics complete.")
