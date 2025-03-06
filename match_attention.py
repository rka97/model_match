from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch

from match_rope import apply_rope_jax, init_jax_rope
from nanodo_model import CausalAttn, DoConfig
from plainlm_model import (
    Attention,
    ModelConfig,
    precompute_freqs_cis,
)


def init_pytorch_attention(dim=256, n_heads=4, seq_len=128):
    """Initialize PyTorch attention module from plainlm_model."""
    print(f"Initializing PyTorch Attention with dim={dim}, n_heads={n_heads}")
    config = ModelConfig(
        vocab_size=1000,  # dummy value
        seq_len=seq_len,  # dummy value
        dim=dim,
        expand=4.0,  # dummy value
        n_layers=1,  # dummy value
        n_heads=n_heads,
        rmsnorm_eps=1e-6,
    )
    # Precompute rotary embeddings
    freqs_cis = precompute_freqs_cis(
        dim // n_heads,
        seq_len,
        theta=500000,  # match plainlm_model's Transformer config
    )
    return Attention(config), freqs_cis


def init_flax_attention(dim=256, n_heads=4, seq_len=128):
    """Initialize Flax attention module from nanodo_model."""
    print(f"Initializing Flax Attention with dim={dim}, n_heads={n_heads}")
    cfg = DoConfig(
        D=dim,
        H=n_heads,
        L=seq_len,
        N=1,  # dummy num layers
        V=1000,  # dummy vocab
        F=1024,  # dummy FF dim
        dtype=jnp.float32,
        rmsnorm_epsilon=1e-6,
    )
    return CausalAttn(cfg)


def copy_attention_params(pytorch_attn, flax_params):
    """Copy parameters from PyTorch Attention to Flax CausalAttn."""
    print("\nCopying attention parameters...")

    n_heads = pytorch_attn.n_heads
    head_dim = pytorch_attn.head_dim
    dim = pytorch_attn.dim
    # Split PyTorch's combined qkv weights
    w_qkv, w_out = pytorch_attn.w_qkv.weight, pytorch_attn.w_out.weight
    print(w_qkv.shape)
    q_weight, k_weight, v_weight = [u.detach().numpy() for u in w_qkv.split(dim, dim=0)]
    print(f"PyTorch parameter shapes before copy:")
    print(f"Query: {q_weight.shape}")
    print(f"Key: {k_weight.shape}")
    print(f"Value: {v_weight.shape}")
    print(f"Output: {pytorch_attn.w_out.weight.shape}")

    # Print parameter shapes for verification
    print("Flax parameter shapes before copy:")
    print(jax.tree.map(lambda x: x.shape, flax_params["params"]))

    # Reshape for Flax's dense general format [D, H, Dh]
    def reshape_for_flax(w, n_heads, head_dim):
        return w.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)  # [D, H, Dh]


    new_params = {
        "query": {
            "kernel": reshape_for_flax(q_weight, n_heads, head_dim)
        },
        "key": {
            "kernel": reshape_for_flax(k_weight, n_heads, head_dim)
        },
        "value": {
            "kernel": reshape_for_flax(v_weight, n_heads, head_dim)
        },
        "attn_out_proj": {
            "kernel": w_out.detach().numpy().T
        },
    }

    # Print parameter shapes for verification
    print("Parameter shapes after copy:")
    for k in new_params:
        print(f"{k}: {new_params[k]['kernel'].shape}")

    return {"params": new_params}


def compare_attention_outputs(dim=256, n_heads=4, seq_len=10, batch_size=2):
    """Compare attention outputs between implementations."""
    # Initialize modules
    torch_attn, freqs_cis = init_pytorch_attention(dim, n_heads, seq_len)
    flax_attn = init_flax_attention(dim, n_heads, seq_len)

    # Initialize Flax params with PyTorch weights
    dummy_input = jnp.ones((batch_size, seq_len, dim))
    flax_params = flax_attn.init(jax.random.PRNGKey(0), dummy_input)
    flax_params = copy_attention_params(torch_attn, flax_params)

    # Generate random input
    np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    torch_input = torch.tensor(np_input)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_attn(torch_input, freqs_cis).numpy()

    # Flax forward pass
    flax_output = flax_attn.apply(flax_params, jnp.array(np_input))
    flax_output = np.array(flax_output)

    print(f"\nOutput shapes:")
    print(f"PyTorch: {torch_output.shape}")
    print(f"Flax: {flax_output.shape}")

    # Calculate differences
    mse = np.mean((torch_output - flax_output)**2)
    max_diff = np.max(np.abs(torch_output - flax_output))

    print(f"\nAttention Comparison Results:")
    print(f"MSE: {mse:.8f}")
    print(f"Max Difference: {max_diff:.8f}")

    return mse, max_diff


def main():
    print("=" * 50)
    print("Comparing Attention Implementations")
    print("=" * 50)

    # Test configuration
    config = {
        "dim": 48,
        "n_heads": 3,
        "seq_len": 16,
        "batch_size": 7,
    }

    # Run comparison
    mse, max_diff = compare_attention_outputs(**config)

    print("\nFinal Results:")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")

    # Run comparison
    mse, max_diff = compare_attention_outputs(**config)

    print("\nFinal Results:")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")


if __name__ == "__main__":
    main()
