import torch
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Import the MLP implementations from both models
from plainlm_model import MLP as PyTorchMLP
from nanodo_model import Mlp as FlaxMLP, DoConfig


def init_pytorch_mlp(dim=256, hidden_dim=1024):
    """Initialize PyTorch MLP module."""
    print(f"Initializing PyTorch MLP with dim={dim}, hidden_dim={hidden_dim}")
    mlp = PyTorchMLP(dim=dim, hidden_dim=hidden_dim)
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


def compare_multiple_inputs(m=20, dim=256, hidden_dim=1024, batch_size=2, seq_len=10):
    """Compare outputs from both MLPs for multiple random inputs."""
    print(f"\nComparing outputs for {m} different random inputs...")

    # Initialize MLPs once for all comparisons
    print("Initializing MLPs once for all comparisons...")
    pytorch_mlp = init_pytorch_mlp(dim, hidden_dim)
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
        pytorch_mlp = init_pytorch_mlp(dim, hidden_dim)
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
    print("Comparing outputs.")
    print("=" * 50)

    # Initialize models once
    print("Initializing models...")
    dim, hidden_dim = 256, 1024
    pytorch_mlp = init_pytorch_mlp(dim, hidden_dim)
    flax_mlp, flax_params = init_flax_mlp(dim, hidden_dim)
    flax_params = copy_pytorch_params_to_flax(pytorch_mlp, flax_params)

    # Run a single comparison with the initialized models
    print("\n" + "=" * 50)
    print("Running single comparison with initialized models")
    print("=" * 50)
    _, _, mse, max_diff = run_single_comparison(
        pytorch_mlp=pytorch_mlp, flax_mlp=flax_mlp, flax_params=flax_params
    )
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")

    # Run comparison for m=20 different inputs
    print("\n" + "=" * 50)
    print("Running multiple comparisons with the same models")
    print("=" * 50)
    avg_mse, avg_max_diff = compare_multiple_inputs(m=20)

    print("\nFinal Results:")
    print(f"Average Mean Squared Error: {avg_mse:.8f}")
    print(f"Average Maximum Absolute Difference: {avg_max_diff:.8f}")


    # Run comparison for m=20 different inputs
    print("\n" + "=" * 50)
    print("Running multiple comparisons with the same models")
    print("=" * 50)
    avg_mse, avg_max_diff = compare_multiple_inputs(m=20)

    print("\nFinal Results:")
    print(f"Average Mean Squared Error: {avg_mse:.8f}")
    print(f"Average Maximum Absolute Difference: {avg_max_diff:.8f}")


if __name__ == "__main__":
    main()
