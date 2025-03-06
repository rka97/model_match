import torch
import jax.numpy as jnp
import numpy as np
from plainlm_model import precompute_freqs_cis, apply_rotary_emb_complex_like
from nanodo_model import init_rope as init_jax_rope, apply_rope as apply_rope_jax

def init_pytorch_rope(dim=256, seq_len=128, n_heads=4):
    """Initialize PyTorch rotary embeddings."""
    print(f"Initializing PyTorch RoPE with dim={dim}, seq_len={seq_len}")
    # Precompute frequencies using PyTorch
    freqs_cis = precompute_freqs_cis(
        dim // n_heads,  # dim per head
        seq_len,
        theta=500000  # match plainlm_model's Transformer config
    )
    return freqs_cis

def compare_rope_outputs(dim=4, seq_len=4, batch_size=2, n_heads=4):
    """Compare rotary embedding outputs between implementations."""
    # Initialize modules
    torch_freqs = init_pytorch_rope(dim, seq_len, n_heads)
    jax_freqs = init_jax_rope(dim, seq_len, n_heads)

    # Generate random input
    head_dim = dim // n_heads
    q_np = np.random.randn(batch_size, seq_len, n_heads, head_dim).astype(np.float32)
    k_np = np.random.randn(batch_size, seq_len, n_heads, head_dim).astype(np.float32)

    # PyTorch forward pass
    q_torch, k_torch = torch.tensor(q_np), torch.tensor(k_np)
    with torch.no_grad():
        q_torch_rot, k_torch_rot = apply_rotary_emb_complex_like(
            q_torch, k_torch, freqs_cis=torch_freqs
        )
        torch_output = q_torch_rot.numpy()

    # JAX forward pass
    q_jax, k_jax = jnp.array(q_np), jnp.array(k_np)
    q_jax_rot, k_jax_rot = apply_rope_jax(q_jax, k_jax, jax_freqs)
    jax_output = np.array(q_jax_rot)

    # Calculate differences
    mse = np.mean((q_torch_rot.numpy() - np.array(q_jax_rot)) ** 2)
    mse += np.mean((k_torch_rot.numpy() - np.array(k_jax_rot)) ** 2)
    max_diff = np.max(np.abs(q_torch_rot.numpy() - np.array(q_jax_rot)))
    max_diff = max(max_diff, np.max(np.abs(k_torch_rot.numpy() - np.array(k_jax_rot))))

    print(f"\nRoPE Comparison Results:")
    print(f"MSE: {mse:.8f}")
    print(f"Max Difference: {max_diff:.8f}")

    return mse, max_diff

def main():
    print("=" * 50)
    print("Comparing Rotary Position Embedding (RoPE) Implementations")
    print("=" * 50)

    # Test configuration
    config = {
        'dim': 128,
        'seq_len': 256,
        'batch_size': 32,
        'n_heads': 32,
    }

    # Run comparison
    mse, max_diff = compare_rope_outputs(**config)

    print("\nFinal Results:")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")

if __name__ == "__main__":
    main()
