import torch
import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

# Import the MLP implementations from both models
from plainlm_model import MLP as PyTorchMLP
from nanodo_model import Mlp as FlaxMLP, DoConfig


def init_pytorch_mlp(dim=256, hidden_dim=1024, compile_model=True):
    """Initialize PyTorch MLP module."""
    print(f"Initializing PyTorch MLP with dim={dim}, hidden_dim={hidden_dim}, compile={compile_model}")
    mlp = PyTorchMLP(dim=dim, hidden_dim=hidden_dim)
    
    # Compile the MLP module if requested
    if compile_model:
        print("Compiling PyTorch MLP with torch.compile...")
        mlp = torch.compile(mlp, fullgraph=True)
    
    return mlp


def init_flax_mlp(dim=256, hidden_dim=1024):
    """Initialize Flax MLP module."""
    print(f"Initializing Flax MLP with dim={dim}, hidden_dim={hidden_dim}")
    # Create a config with the same dimensions
    cfg = DoConfig(
        D=dim,  # model dimension
        H=4,  # not used by MLP
        L=128,  # not used by MLP
        N=2,  # not used by MLP
        V=1000,  # not used by MLP
        F=hidden_dim,  # hidden dimension
        dtype=jnp.float32,
        rmsnorm_epsilon=1e-6,
    )

    # Initialize the MLP
    mlp = FlaxMLP(cfg)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 10, dim))  # batch_size=1, seq_len=10
    params = mlp.init(rng, dummy_input)

    return mlp, params


def copy_pytorch_params_to_flax(pytorch_mlp, flax_params):
    """
    Copy parameters from PyTorch MLP to Flax MLP.

    Args:
        pytorch_mlp: PyTorch MLP model containing fc1 and fc2 layers
        flax_params: Flax parameter dictionary to be updated

    Returns:
        Updated Flax parameter dictionary
    """
    print("\nCopying PyTorch parameters to Flax MLP...")

    # Handle compiled models
    if hasattr(pytorch_mlp, '_orig_mod'):
        pytorch_mlp = pytorch_mlp._orig_mod

    # Create a new params dict to avoid modifying the original
    new_params = flax_params.copy()

    # Define layer mapping between PyTorch and Flax
    layer_mapping = {"fc1": "Dense_0", "fc2": "Dense_1"}

    # Copy parameters for each layer
    for pytorch_name, flax_name in layer_mapping.items():
        if hasattr(pytorch_mlp, pytorch_name):
            # Extract PyTorch weights
            pytorch_weight = getattr(pytorch_mlp, pytorch_name).weight.detach().numpy()

            # In Flax, the weights are transposed compared to PyTorch
            # PyTorch: [out_features, in_features]
            # Flax: [in_features, out_features]
            flax_weight = pytorch_weight.T

            # Update the weights in the new params dict
            if flax_name in new_params["params"]:
                new_params["params"][flax_name]["kernel"] = flax_weight
            else:
                print(f"Warning: {flax_name} not found in Flax params")

    print("Parameters copied successfully!")
    return new_params


def compare_mlp_outputs(dim=256, hidden_dim=1024, batch_size=2, seq_len=10, num_trials=100, compile_pytorch=True):
    """Compare MLP outputs and timing between implementations."""
    print(f"\nComparing MLPs with {num_trials} trials...")
    
    # Initialize MLPs
    pytorch_mlp = init_pytorch_mlp(dim, hidden_dim, compile_model=compile_pytorch)
    flax_mlp, flax_params = init_flax_mlp(dim, hidden_dim)
    flax_params = copy_pytorch_params_to_flax(pytorch_mlp, flax_params)

    # Generate random input for initial comparison
    np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    torch_input = torch.tensor(np_input)
    jax_input = jnp.array(np_input)

    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = pytorch_mlp(torch_input).numpy()

    # Flax forward pass
    flax_output = np.array(flax_mlp.apply(flax_params, jax_input))

    print(f"\nOutput shapes:")
    print(f"PyTorch: {pytorch_output.shape}")
    print(f"Flax: {flax_output.shape}")

    # Calculate differences
    mse = np.mean((pytorch_output - flax_output)**2)
    max_diff = np.max(np.abs(pytorch_output - flax_output))

    print(f"\nMLP Comparison Results:")
    print(f"MSE: {mse:.8f}")
    print(f"Max Difference: {max_diff:.8f}")

    # Timing comparison
    print(f"\n{'=' * 50}")
    print(f"Running timing comparison with {num_trials} trials...")
    print(f"PyTorch compiled: {compile_pytorch}")
    print(f"{'=' * 50}")

    torch_times = []
    flax_times = []

    # Warmup runs
    print("Warming up (this may take longer for compiled models)...")
    for _ in range(10):
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        with torch.no_grad():
            _ = pytorch_mlp(torch.tensor(np_input))
        _ = flax_mlp.apply(flax_params, jnp.array(np_input))
    
    # JAX block_until_ready to ensure compilation is complete
    jax.block_until_ready(flax_mlp.apply(flax_params, jnp.array(np_input)))

    print(f"Starting {num_trials} timed trials...")
    for i in range(num_trials):
        # Generate new random input for each trial
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        torch_input = torch.tensor(np_input)
        jax_input = jnp.array(np_input)

        # Time PyTorch
        torch_start = time.perf_counter()
        with torch.no_grad():
            _ = pytorch_mlp(torch_input)
        torch_end = time.perf_counter()
        torch_times.append(torch_end - torch_start)

        # Time JAX/Flax
        flax_start = time.perf_counter()
        flax_result = flax_mlp.apply(flax_params, jax_input)
        jax.block_until_ready(flax_result)  # Ensure computation is complete
        flax_end = time.perf_counter()
        flax_times.append(flax_end - flax_start)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_trials} trials")

    # Convert to arrays for statistics
    torch_times = np.array(torch_times) * 1000  # Convert to milliseconds
    flax_times = np.array(flax_times) * 1000

    print(f"\n{'=' * 50}")
    print("Timing Results:")
    print(f"{'=' * 50}")
    print(f"\nPyTorch MLP (compiled={compile_pytorch}):")
    print(f"  Mean: {np.mean(torch_times):.4f} ms")
    print(f"  Median: {np.median(torch_times):.4f} ms")
    print(f"  Std Dev: {np.std(torch_times):.4f} ms")
    print(f"  Min: {np.min(torch_times):.4f} ms")
    print(f"  Max: {np.max(torch_times):.4f} ms")

    print(f"\nJAX/Flax MLP:")
    print(f"  Mean: {np.mean(flax_times):.4f} ms")
    print(f"  Median: {np.median(flax_times):.4f} ms")
    print(f"  Std Dev: {np.std(flax_times):.4f} ms")
    print(f"  Min: {np.min(flax_times):.4f} ms")
    print(f"  Max: {np.max(flax_times):.4f} ms")

    speedup = np.mean(torch_times) / np.mean(flax_times)
    print(f"\nSpeedup (PyTorch/JAX): {speedup:.2f}x")
    if speedup > 1:
        print(f"JAX is {speedup:.2f}x faster than PyTorch")
    else:
        print(f"PyTorch is {1/speedup:.2f}x faster than JAX")

    return mse, max_diff, torch_times, flax_times


def compare_multiple_inputs(m=20, dim=256, hidden_dim=1024, batch_size=2, seq_len=10):
    """Compare outputs from both MLPs for multiple random inputs."""
    print(f"\nComparing outputs for {m} different random inputs...")

    # Initialize MLPs once for all comparisons
    print("Initializing MLPs once for all comparisons...")
    pytorch_mlp = init_pytorch_mlp(dim, hidden_dim, compile_model=True)
    flax_mlp, flax_params = init_flax_mlp(dim, hidden_dim)

    # Copy parameters
    flax_params = copy_pytorch_params_to_flax(pytorch_mlp, flax_params)

    # Track errors across all runs
    all_mse = []
    all_max_diff = []

    # Run comparison for m different inputs
    for i in range(m):
        # Create identical random inputs
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        torch_input = torch.tensor(np_input)
        jax_input = jnp.array(np_input)

        # Get outputs
        with torch.no_grad():
            pytorch_output = pytorch_mlp(torch_input).numpy()

        flax_output = np.array(flax_mlp.apply(flax_params, jax_input))

        # Calculate MSE
        mse = np.mean((pytorch_output - flax_output) ** 2)
        all_mse.append(mse)

        # Calculate max absolute difference
        max_diff = np.max(np.abs(pytorch_output - flax_output))
        all_max_diff.append(max_diff)

        # Print progress
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{m} comparisons...")

    # Calculate statistics across all runs
    avg_mse = np.mean(all_mse)
    std_mse = np.std(all_mse)
    min_mse = np.min(all_mse)
    max_mse = np.max(all_mse)

    avg_max_diff = np.mean(all_max_diff)
    std_max_diff = np.std(all_max_diff)
    min_max_diff = np.min(all_max_diff)
    max_max_diff = np.max(all_max_diff)

    # Print summary statistics
    print("\nSummary Statistics across all runs:")
    print(f"  Mean Squared Error:")
    print(f"    Average: {avg_mse:.8f}")
    print(f"    Std Dev: {std_mse:.8f}")
    print(f"    Min: {min_mse:.8f}")
    print(f"    Max: {max_mse:.8f}")

    print(f"\n  Maximum Absolute Difference:")
    print(f"    Average: {avg_max_diff:.8f}")
    print(f"    Std Dev: {std_max_diff:.8f}")
    print(f"    Min: {min_max_diff:.8f}")
    print(f"    Max: {max_max_diff:.8f}")

    return avg_mse, avg_max_diff


def run_single_comparison(
    pytorch_mlp=None,
    flax_mlp=None,
    flax_params=None,
    dim=256,
    hidden_dim=1024,
    batch_size=2,
    seq_len=10,
):
    """Run a single comparison between PyTorch and Flax MLPs."""
    # Initialize MLPs if not provided
    if pytorch_mlp is None or flax_mlp is None or flax_params is None:
        pytorch_mlp = init_pytorch_mlp(dim, hidden_dim, compile_model=True)
        flax_mlp, flax_params = init_flax_mlp(dim, hidden_dim)
        flax_params = copy_pytorch_params_to_flax(pytorch_mlp, flax_params)

    # Create identical random inputs
    np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    torch_input = torch.tensor(np_input)
    jax_input = jnp.array(np_input)

    # Get outputs
    with torch.no_grad():
        pytorch_output = pytorch_mlp(torch_input).numpy()

    flax_output = np.array(flax_mlp.apply(flax_params, jax_input))

    # Calculate MSE
    mse = np.mean((pytorch_output - flax_output) ** 2)

    # Calculate max absolute difference
    max_diff = np.max(np.abs(pytorch_output - flax_output))

    return pytorch_output, flax_output, mse, max_diff


def main():
    print("=" * 50)
    print("Comparing MLP Implementations")
    print("=" * 50)

    # Test configuration
    config = {
        "dim": 768,
        "hidden_dim": 3072,
        "seq_len": 1024,
        "batch_size": 16,
        "num_trials": 100,
        "compile_pytorch": True,
    }

    # Run comparison with timing
    mse, max_diff, torch_times, flax_times = compare_mlp_outputs(**config)

    print(f"\n{'=' * 50}")
    print("Final Summary:")
    print(f"{'=' * 50}")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")
    print(f"PyTorch Mean Time: {np.mean(torch_times):.4f} ms")
    print(f"JAX Mean Time: {np.mean(flax_times):.4f} ms")


if __name__ == "__main__":
    main()
