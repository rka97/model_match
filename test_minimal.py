"""
Minimal performance comparison of elementary operations between PyTorch and JAX.
Tests basic operations to identify if speed differences are inherent to the frameworks.
"""

import time
import numpy as np
import torch
import jax
import jax.numpy as jnp


# Global dtype setting
USE_BFLOAT16 = False  # Set to True to use bfloat16 instead of float32


def get_dtype():
    """Get the current dtype for both frameworks."""
    if USE_BFLOAT16:
        return torch.bfloat16, jnp.bfloat16, np.float32  # NumPy doesn't have bfloat16, use float32 for initialization
    else:
        return torch.float32, jnp.float32, np.float32


def benchmark_operation(name, torch_fn, jax_fn, num_trials=100, warmup=10):
    """
    Benchmark a specific operation in both PyTorch and JAX.
    
    Args:
        name: Name of the operation
        torch_fn: PyTorch operation function
        jax_fn: JAX operation function
        num_trials: Number of timing trials
        warmup: Number of warmup runs
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        _ = torch_fn()
        result = jax_fn()
        jax.block_until_ready(result)
    
    # Benchmark PyTorch
    torch_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        _ = torch_fn()
        end = time.perf_counter()
        torch_times.append(end - start)
    
    # Benchmark JAX
    jax_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        result = jax_fn()
        jax.block_until_ready(result)
        end = time.perf_counter()
        jax_times.append(end - start)
    
    # Convert to milliseconds
    torch_times = np.array(torch_times) * 1000
    jax_times = np.array(jax_times) * 1000
    
    # Print results
    print(f"\nPyTorch (compiled):")
    print(f"  Mean: {np.mean(torch_times):.4f} ms")
    print(f"  Median: {np.median(torch_times):.4f} ms")
    print(f"  Std Dev: {np.std(torch_times):.4f} ms")
    
    print(f"\nJAX (JIT):")
    print(f"  Mean: {np.mean(jax_times):.4f} ms")
    print(f"  Median: {np.median(jax_times):.4f} ms")
    print(f"  Std Dev: {np.std(jax_times):.4f} ms")
    
    speedup = np.mean(torch_times) / np.mean(jax_times)
    print(f"\nSpeedup (PyTorch/JAX): {speedup:.2f}x")
    if speedup > 1:
        print(f"JAX is {speedup:.2f}x faster")
    else:
        print(f"PyTorch is {1/speedup:.2f}x faster")
    
    return torch_times, jax_times


def test_matmul(M=1024, K=768, N=768):
    """Test matrix multiplication: (M, K) @ (K, N) -> (M, N)"""
    print(f"\nMatrix shapes: ({M}, {K}) @ ({K}, {N})")
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random matrices
    np_a = np.random.randn(M, K).astype(np_dtype)
    np_b = np.random.randn(K, N).astype(np_dtype)
    
    torch_a = torch.from_numpy(np_a).to(torch_dtype)
    torch_b = torch.from_numpy(np_b).to(torch_dtype)
    
    jax_a = jnp.array(np_a, dtype=jax_dtype)
    jax_b = jnp.array(np_b, dtype=jax_dtype)
    
    # Compile PyTorch operations
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_matmul_op(a, b):
        return torch.matmul(a, b)
    
    @jax.jit
    def jax_matmul_op(a, b):
        return jnp.matmul(a, b)
    
    torch_fn = lambda: torch_matmul_op(torch_a, torch_b)
    jax_fn = lambda: jax_matmul_op(jax_a, jax_b)
    
    return benchmark_operation("Matrix Multiplication", torch_fn, jax_fn)


def test_batched_matmul(B=16, M=1024, K=768, N=768):
    """Test batched matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N)"""
    print(f"\nBatched matrix shapes: ({B}, {M}, {K}) @ ({B}, {K}, {N})")
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random matrices
    np_a = np.random.randn(B, M, K).astype(np_dtype)
    np_b = np.random.randn(B, K, N).astype(np_dtype)
    
    torch_a = torch.from_numpy(np_a).to(torch_dtype)
    torch_b = torch.from_numpy(np_b).to(torch_dtype)
    
    jax_a = jnp.array(np_a, dtype=jax_dtype)
    jax_b = jnp.array(np_b, dtype=jax_dtype)
    
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_batched_matmul(a, b):
        return torch.matmul(a, b)
    
    @jax.jit
    def jax_batched_matmul(a, b):
        return jnp.matmul(a, b)
    
    torch_fn = lambda: torch_batched_matmul(torch_a, torch_b)
    jax_fn = lambda: jax_batched_matmul(jax_a, jax_b)
    
    return benchmark_operation("Batched Matrix Multiplication", torch_fn, jax_fn)


def test_einsum_attention(B=16, H=12, L=1024, D=64):
    """Test attention-like einsum: bhqd,bhkd->bhqk"""
    print(f"\nEinsum shapes: ({B}, {H}, {L}, {D}), ({B}, {H}, {L}, {D}) -> ({B}, {H}, {L}, {L})")
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random tensors
    np_q = np.random.randn(B, H, L, D).astype(np_dtype)
    np_k = np.random.randn(B, H, L, D).astype(np_dtype)
    
    torch_q = torch.from_numpy(np_q).to(torch_dtype)
    torch_k = torch.from_numpy(np_k).to(torch_dtype)
    
    jax_q = jnp.array(np_q, dtype=jax_dtype)
    jax_k = jnp.array(np_k, dtype=jax_dtype)
    
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_einsum_scores(q, k):
        return torch.einsum('bhqd,bhkd->bhqk', q, k)
    
    @jax.jit
    def jax_einsum_scores(q, k):
        return jnp.einsum('bhqd,bhkd->bhqk', q, k)
    
    torch_fn = lambda: torch_einsum_scores(torch_q, torch_k)
    jax_fn = lambda: jax_einsum_scores(jax_q, jax_k)
    
    return benchmark_operation("Einsum (Attention Scores)", torch_fn, jax_fn)


def test_einsum_values(B=16, H=12, L=1024, D=64):
    """Test attention output einsum: bhqk,bhkd->bhqd"""
    print(f"\nEinsum shapes: ({B}, {H}, {L}, {L}), ({B}, {H}, {L}, {D}) -> ({B}, {H}, {L}, {D})")
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random tensors
    np_att = np.random.randn(B, H, L, L).astype(np_dtype)
    np_v = np.random.randn(B, H, L, D).astype(np_dtype)
    
    torch_att = torch.from_numpy(np_att).to(torch_dtype)
    torch_v = torch.from_numpy(np_v).to(torch_dtype)
    
    jax_att = jnp.array(np_att, dtype=jax_dtype)
    jax_v = jnp.array(np_v, dtype=jax_dtype)
    
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_einsum_output(att, v):
        return torch.einsum('bhqk,bhkd->bhqd', att, v)
    
    @jax.jit
    def jax_einsum_output(att, v):
        return jnp.einsum('bhqk,bhkd->bhqd', att, v)
    
    torch_fn = lambda: torch_einsum_output(torch_att, torch_v)
    jax_fn = lambda: jax_einsum_output(jax_att, jax_v)
    
    return benchmark_operation("Einsum (Attention Output)", torch_fn, jax_fn)


def test_layer_norm(B=16, L=1024, D=768):
    """Test layer normalization"""
    print(f"\nLayer norm shape: ({B}, {L}, {D})")
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random tensor
    np_x = np.random.randn(B, L, D).astype(np_dtype)
    
    torch_x = torch.from_numpy(np_x).to(torch_dtype)
    jax_x = jnp.array(np_x, dtype=jax_dtype)
    
    # PyTorch LayerNorm with compilation
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_layer_norm_op(x):
        # Manual layer norm to ensure it's fully compiled
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6)
    
    # JAX manual layer norm
    @jax.jit
    def jax_layer_norm(x, eps=1e-6):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + eps)
    
    torch_fn = lambda: torch_layer_norm_op(torch_x)
    jax_fn = lambda: jax_layer_norm(jax_x)
    
    return benchmark_operation("Layer Normalization", torch_fn, jax_fn)


def test_softmax(B=16, H=12, L=1024):
    """Test softmax operation"""
    print(f"\nSoftmax shape: ({B}, {H}, {L}, {L})")
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random tensor
    np_x = np.random.randn(B, H, L, L).astype(np_dtype)
    
    torch_x = torch.from_numpy(np_x).to(torch_dtype)
    jax_x = jnp.array(np_x, dtype=jax_dtype)
    
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_softmax_op(x):
        return torch.softmax(x, dim=-1)
    
    @jax.jit
    def jax_softmax_op(x):
        return jax.nn.softmax(x, axis=-1)
    
    torch_fn = lambda: torch_softmax_op(torch_x)
    jax_fn = lambda: jax_softmax_op(jax_x)
    
    return benchmark_operation("Softmax", torch_fn, jax_fn)


def test_gelu(B=16, L=1024, D=3072):
    """Test GELU activation"""
    print(f"\nGELU shape: ({B}, {L}, {D})")
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random tensor
    np_x = np.random.randn(B, L, D).astype(np_dtype)
    
    torch_x = torch.from_numpy(np_x).to(torch_dtype)
    jax_x = jnp.array(np_x, dtype=jax_dtype)
    
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_gelu_op(x):
        return torch.nn.functional.gelu(x)
    
    @jax.jit
    def jax_gelu_op(x):
        return jax.nn.gelu(x)
    
    torch_fn = lambda: torch_gelu_op(torch_x)
    jax_fn = lambda: jax_gelu_op(jax_x)
    
    return benchmark_operation("GELU Activation", torch_fn, jax_fn)


def test_elementwise_ops(B=16, L=1024, D=768):
    """Test element-wise operations (add, multiply, sqrt)"""
    print(f"\nElement-wise ops shape: ({B}, {L}, {D})")
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random tensors
    np_a = np.random.randn(B, L, D).astype(np_dtype)
    np_b = np.random.randn(B, L, D).astype(np_dtype)
    
    torch_a = torch.from_numpy(np_a).to(torch_dtype)
    torch_b = torch.from_numpy(np_b).to(torch_dtype)
    
    jax_a = jnp.array(np_a, dtype=jax_dtype)
    jax_b = jnp.array(np_b, dtype=jax_dtype)
    
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_elementwise(a, b):
        return torch.sqrt(a * b + 1.0)
    
    @jax.jit
    def jax_elementwise(a, b):
        return jnp.sqrt(a * b + 1.0)
    
    torch_fn = lambda: torch_elementwise(torch_a, torch_b)
    jax_fn = lambda: jax_elementwise(jax_a, jax_b)
    
    return benchmark_operation("Element-wise Operations", torch_fn, jax_fn)


def test_with_compilation():
    """Test operations with compilation enabled"""
    print("\n" + "=" * 60)
    print("TESTING COMPILED COMPOSITE OPERATION")
    print("=" * 60)
    
    B, L, D = 16, 1024, 768
    
    torch_dtype, jax_dtype, np_dtype = get_dtype()
    
    # Create random tensors
    np_a = np.random.randn(B, L, D).astype(np_dtype)
    np_b = np.random.randn(B, L, D).astype(np_dtype)
    
    torch_a = torch.from_numpy(np_a).to(torch_dtype)
    torch_b = torch.from_numpy(np_b).to(torch_dtype)
    
    jax_a = jnp.array(np_a, dtype=jax_dtype)
    jax_b = jnp.array(np_b, dtype=jax_dtype)
    
    # Define composite operation with max optimization
    @torch.compile(mode="max-autotune", fullgraph=True)
    def torch_composite(a, b):
        c = torch.matmul(a, b.transpose(-1, -2))
        c = torch.softmax(c, dim=-1)
        c = torch.matmul(c, b)
        return c
    
    @jax.jit
    def jax_composite(a, b):
        c = jnp.matmul(a, jnp.swapaxes(b, -1, -2))
        c = jax.nn.softmax(c, axis=-1)
        c = jnp.matmul(c, b)
        return c
    
    torch_fn = lambda: torch_composite(torch_a, torch_b)
    jax_fn = lambda: jax_composite(jax_a, jax_b)
    
    return benchmark_operation("Compiled Composite Operation", torch_fn, jax_fn)


def main(use_bfloat16=False):
    global USE_BFLOAT16
    USE_BFLOAT16 = use_bfloat16
    
    dtype_str = "bfloat16" if use_bfloat16 else "float32"
    
    print("=" * 60)
    print("MINIMAL PYTORCH VS JAX PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"\nTesting elementary operations to identify performance differences")
    print(f"Precision: {dtype_str}")
    print(f"PyTorch: torch.compile with mode='max-autotune', fullgraph=True")
    print(f"JAX: @jax.jit")
    
    # Store all results
    results = {}
    
    # Test basic operations
    print("\n" + "=" * 60)
    print("BASIC OPERATIONS")
    print("=" * 60)
    
    results['matmul'] = test_matmul(M=1024, K=768, N=768)
    results['batched_matmul'] = test_batched_matmul(B=16, M=1024, K=768, N=768)
    results['einsum_scores'] = test_einsum_attention(B=16, H=12, L=1024, D=64)
    results['einsum_output'] = test_einsum_values(B=16, H=12, L=1024, D=64)
    results['layer_norm'] = test_layer_norm(B=16, L=1024, D=768)
    results['softmax'] = test_softmax(B=16, H=12, L=1024)
    results['gelu'] = test_gelu(B=16, L=1024, D=3072)
    results['elementwise'] = test_elementwise_ops(B=16, L=1024, D=768)
    
    # Test with compilation
    results['compiled'] = test_with_compilation()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY ({dtype_str})")
    print("=" * 60)
    
    for name, (torch_times, jax_times) in results.items():
        speedup = np.mean(torch_times) / np.mean(jax_times)
        winner = "JAX" if speedup > 1 else "PyTorch"
        print(f"{name:25s}: {winner:8s} is {abs(speedup):.2f}x faster "
              f"(PyTorch: {np.mean(torch_times):.4f}ms, JAX: {np.mean(jax_times):.4f}ms)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs JAX performance")
    parser.add_argument("--bfloat16", action="store_true", help="Use bfloat16 precision instead of float32")
    args = parser.parse_args()
    
    main(use_bfloat16=args.bfloat16)