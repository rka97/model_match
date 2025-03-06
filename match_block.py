import jax
import jax.numpy as jnp
import numpy as np
import torch

from nanodo_model import TBlock, DoConfig
from plainlm_model import Block, ModelConfig, precompute_freqs_cis


def init_pytorch_block(dim=256, n_heads=4, expand=4.0, seq_len=128):
    """Initialize PyTorch Block from plainlm_model."""
    print(f"Initializing PyTorch Block with dim={dim}, n_heads={n_heads}, seq_len={seq_len}")
    config = ModelConfig(
        vocab_size=1000,  # dummy value
        seq_len=seq_len,
        dim=dim,
        expand=expand,
        n_layers=1,  # dummy value
        n_heads=n_heads,
        rmsnorm_eps=1e-6,
    )
    return Block(layer_id=0, cfg=config)


def init_flax_block(dim=256, n_heads=4, expand=4.0, seq_len=128):
    """Initialize Flax TBlock from nanodo_model."""
    print(f"Initializing Flax TBlock with dim={dim}, n_heads={n_heads}, seq_len={seq_len}")
    cfg = DoConfig(
        D=dim,
        H=n_heads,
        L=seq_len,
        N=1,  # dummy num layers
        V=1000,  # dummy vocab
        F=int(dim * expand),  # FF dim based on expand factor
        dtype=jnp.float32,
        rmsnorm_epsilon=1e-6,
    )
    return TBlock(cfg)


def copy_block_params(pytorch_block, flax_params):
    """Copy parameters from PyTorch Block to Flax TBlock."""
    print("\nCopying block parameters...")

    # Copy attention parameters
    attn_params = {}
    pytorch_attn = pytorch_block.attn
    n_heads = pytorch_attn.n_heads
    head_dim = pytorch_attn.head_dim
    dim = pytorch_attn.dim

    # Split PyTorch's combined qkv weights
    w_qkv, w_out = pytorch_attn.w_qkv.weight, pytorch_attn.w_out.weight
    q_weight, k_weight, v_weight = [u.detach().numpy() for u in w_qkv.split(dim, dim=0)]

    # Reshape for Flax's dense general format [D, H, Dh]
    def reshape_for_flax(w, n_heads, head_dim):
        return w.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)

    attn_params = {
        "query": {"kernel": reshape_for_flax(q_weight, n_heads, head_dim)},
        "key": {"kernel": reshape_for_flax(k_weight, n_heads, head_dim)},
        "value": {"kernel": reshape_for_flax(v_weight, n_heads, head_dim)},
        "attn_out_proj": {"kernel": w_out.detach().numpy().T},
    }

    # Copy MLP parameters
    pytorch_mlp = pytorch_block.mlp
    mlp_params = {
        "Dense_0": {"kernel": pytorch_mlp.fc1.weight.detach().numpy().T},
        "Dense_1": {"kernel": pytorch_mlp.fc2.weight.detach().numpy().T},
    }

    # Copy RMSNorm parameters
    norm_params = {
        "attn_norm": {
            "scale": pytorch_block.attn_norm.weight.detach().numpy(),
        },
        "mlp_norm": {
            "scale": pytorch_block.mlp_norm.weight.detach().numpy(),
        },
    }

    return {
        "params": {
            "CausalAttn_0": attn_params,
            "Mlp_0": mlp_params,
            "RMSNorm_0": norm_params["attn_norm"],
            "RMSNorm_1": norm_params["mlp_norm"],
        }
    }


def compare_block_outputs(dim=256, n_heads=4, seq_len=10, batch_size=2, expand=4.0):
    """Compare block outputs between implementations."""
    # Initialize modules
    torch_block = init_pytorch_block(dim, n_heads, expand, seq_len)
    flax_block = init_flax_block(dim, n_heads, expand, seq_len)

    # Initialize Flax params with PyTorch weights
    dummy_input = jnp.ones((batch_size, seq_len, dim))
    flax_params = flax_block.init(jax.random.PRNGKey(0), dummy_input)
    print(jax.tree.map(lambda x: x.shape, flax_params["params"]))
    flax_params = copy_block_params(torch_block, flax_params)

    # Generate random input
    np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    torch_input = torch.tensor(np_input)

    # Precompute rotary embeddings for PyTorch
    freqs_cis = precompute_freqs_cis(dim // n_heads, seq_len, theta=500000)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_block(torch_input, freqs_cis).numpy()

    # Flax forward pass
    flax_output = flax_block.apply(flax_params, jnp.array(np_input))
    flax_output = np.array(flax_output)

    print(f"\nOutput shapes:")
    print(f"PyTorch: {torch_output.shape}")
    print(f"Flax: {flax_output.shape}")

    # Calculate differences
    mse = np.mean((torch_output - flax_output)**2)
    max_diff = np.max(np.abs(torch_output - flax_output))

    print(f"\nBlock Comparison Results:")
    print(f"MSE: {mse:.8f}")
    print(f"Max Difference: {max_diff:.8f}")

    return mse, max_diff


def main():
    print("=" * 50)
    print("Comparing Transformer Block Implementations")
    print("=" * 50)

    # Test configuration
    config = {
        "dim": 48,
        "n_heads": 3,
        "seq_len": 16,
        "batch_size": 7,
        "expand": 4.0,
    }

    # Run comparison
    mse, max_diff = compare_block_outputs(**config)

    print("\nFinal Results:")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")


if __name__ == "__main__":
    main()
