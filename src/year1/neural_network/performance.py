"""
Phase 1.4 - Performance Analysis
Covers: manual loops vs NumPy benchmarking, profiling bottlenecks
"""

import numpy as np
import time
import cProfile
import pstats
import io


def manual_matmul(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C


def manual_relu(x):
    return [max(0, v) for v in x]


def manual_mse(y_pred, y_true):
    n = len(y_pred)
    return sum((y_pred[i] - y_true[i]) ** 2 for i in range(n)) / n


def numpy_matmul(A, B):
    return A @ B


def numpy_relu(x):
    return np.maximum(0, x)


def numpy_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def benchmark(fn, *args, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times), result


if __name__ == "__main__":
    print("=== MATRIX MULTIPLICATION BENCHMARK ===")
    SIZE = 100
    A_list = [[float(i + j) for j in range(SIZE)] for i in range(SIZE)]
    B_list = [[float(i * j + 1) for j in range(SIZE)] for i in range(SIZE)]
    A_np = np.array(A_list)
    B_np = np.array(B_list)

    manual_time, _ = benchmark(manual_matmul, A_list, B_list)
    numpy_time,  _ = benchmark(numpy_matmul,  A_np,   B_np)
    print(f"Size: {SIZE}x{SIZE}")
    print(f"Manual: {manual_time:.4f}s")
    print(f"NumPy:  {numpy_time:.6f}s")
    print(f"Speedup: {manual_time / numpy_time:.1f}x")

    print("\n=== ACTIVATION FUNCTION BENCHMARK ===")
    N = 100_000
    x_list = [float(i) / 1000 - 50 for i in range(N)]
    x_np   = np.array(x_list)

    manual_time, _ = benchmark(manual_relu, x_list)
    numpy_time,  _ = benchmark(numpy_relu,  x_np)
    print(f"Elements: {N:,}")
    print(f"Manual ReLU: {manual_time:.4f}s")
    print(f"NumPy  ReLU: {numpy_time:.6f}s")
    print(f"Speedup: {manual_time / numpy_time:.1f}x")

    print("\n=== MSE BENCHMARK ===")
    y_pred_list = [float(i) / N for i in range(N)]
    y_true_list = [float(i + 1) / N for i in range(N)]
    y_pred_np   = np.array(y_pred_list)
    y_true_np   = np.array(y_true_list)

    manual_time, _ = benchmark(manual_mse, y_pred_list, y_true_list)
    numpy_time,  _ = benchmark(numpy_mse,  y_pred_np,   y_true_np)
    print(f"Elements: {N:,}")
    print(f"Manual MSE: {manual_time:.4f}s")
    print(f"NumPy  MSE: {numpy_time:.6f}s")
    print(f"Speedup: {manual_time / numpy_time:.1f}x")

    print("\n=== PROFILING NEURAL NETWORK FORWARD PASS ===")
    from src.year1.neural_network.network import NeuralNetwork
    np.random.seed(42)
    nn = NeuralNetwork([2, 16, 8, 1], ['relu', 'relu', 'sigmoid'])
    X  = np.random.randn(1000, 2)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(100):
        nn.forward(X)
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats('cumulative')
    ps.print_stats(8)
    print(stream.getvalue())

    print("Done: Performance analysis complete.")
