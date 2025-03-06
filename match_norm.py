import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

def compare_rmsnorm(dim=256, batch_size=2, seq_len=10, eps=1e-6):
    """Compare PyTorch and Flax RMSNorm implementations."""
    # Initialize modules
    torch_norm = torch.nn.RMSNorm(dim, eps=eps)
    flax_norm = nn.RMSNorm(epsilon=eps)

    # Initialize Flax params
    dummy_input = jnp.ones((batch_size, seq_len, dim))
    flax_params = flax_norm.init(jax.random.PRNGKey(0), dummy_input)

    # Initialize random weights for Flax
    rng = jax.random.PRNGKey(42)
    random_weight = jax.random.normal(rng, (dim,)) * 0.02  # Small random weights
    flax_params['params']['scale'] = random_weight

    # Also set PyTorch weights to same values for fair comparison
    with torch.no_grad():
        torch_norm.weight.copy_(torch.tensor(np.array(random_weight)))

    # Generate random input
    np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    torch_input = torch.tensor(np_input)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_norm(torch_input).numpy()

    # Flax forward pass
    flax_output = flax_norm.apply(flax_params, jnp.array(np_input))
    flax_output = np.array(flax_output)

    # Calculate differences
    mse = np.mean((torch_output - flax_output) ** 2)
    max_diff = np.max(np.abs(torch_output - flax_output))

    print(f"\nRMSNorm Comparison Results:")
    print(f"Input shape: {np_input.shape}")
    print(f"MSE: {mse:.8f}")
    print(f"Max Difference: {max_diff:.8f}")

    return mse, max_diff

def main():
    print("=" * 50)
    print("Comparing RMSNorm Implementations")
    print("=" * 50)
    
    # Test configuration
    config = {
        'dim': 256,
        'batch_size': 2,
        'seq_len': 10,
        'eps': 1e-6,
    }
    
    # Run comparison
    mse, max_diff = compare_rmsnorm(**config)
    
    print("\nFinal Results:")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")

if __name__ == "__main__":
    main()
