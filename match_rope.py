import torch
import jax
import jax.numpy as jnp
import numpy as np
from plainlm_model import precompute_freqs_cis, apply_rotary_emb_complex_like
import functools

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

@functools.partial(jax.jit, static_argnums=(0,1,2))
def init_jax_rope(dim=256, seq_len=128, n_heads=4):
    """Initialize JAX rotary embeddings."""
    print(f"Initializing JAX RoPE with dim={dim}, seq_len={seq_len}")

    # JAX implementation of precompute_freqs_cis
    def precompute_freqs_cis_jax(dim, end, theta=10000.0):
        inv_freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
        t = jnp.arange(end) / 1.0
        freqs = jnp.outer(t, inv_freqs).astype(jnp.float32)
        return jnp.stack([
            jnp.cos(freqs)[None, :, None, :],
            jnp.sin(freqs)[None, :, None, :]
        ], axis=3)

    freqs_cis = precompute_freqs_cis_jax(dim // n_heads, seq_len, theta=500000)
    return freqs_cis.transpose(0, 1, 2, 4, 3)

@jax.jit
def apply_rope_jax(q, k, freqs_cis):
    """Apply rotary embeddings in JAX to Q and K separately."""
    def rotate_tensor(x):
        # Split into real and imaginary parts
        x_r2 = x.reshape(*x.shape[:-1], -1, 2)

        # Apply rotation
        rotated_x_r2 = jnp.stack([
            x_r2[..., 0] * freqs_cis[..., 0] - x_r2[..., 1] * freqs_cis[..., 1],
            x_r2[..., 1] * freqs_cis[..., 0] + x_r2[..., 0] * freqs_cis[..., 1]
        ], axis=-1)

        return rotated_x_r2.reshape(*x.shape)

    # Apply rotation to Q and K separately
    rotated_q = rotate_tensor(q)
    rotated_k = rotate_tensor(k)

    return rotated_q, rotated_k

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
