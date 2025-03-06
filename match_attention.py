import torch
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from plainlm_model import Attention as PyTorchAttention, ModelConfig, apply_rotary_emb_complex_like, precompute_freqs_cis
from nanodo_model import CausalAttn, DoConfig
from match_rope import apply_rope_jax, init_jax_rope

def init_pytorch_attention(dim=256, n_heads=4):
    """Initialize PyTorch attention module."""
    print(f"Initializing PyTorch Attention with dim={dim}, n_heads={n_heads}")
    config = ModelConfig(
        vocab_size=1000,  # dummy value
        seq_len=128,      # dummy value
        dim=dim,
        expand=4.0,       # dummy value
        n_layers=1,       # dummy value
        n_heads=n_heads,
    )
    return PyTorchAttention(config)

def init_flax_attention(dim=256, n_heads=4):
    """Initialize Flax attention module."""
    print(f"Initializing Flax Attention with dim={dim}, n_heads={n_heads}")
    cfg = DoConfig(
        D=dim,
        H=n_heads,
        L=128,  # dummy sequence length
        N=1,    # dummy num layers
        V=1000,  # dummy vocab
        F=1024,  # dummy FF dim
    )
    return CausalAttn(cfg)

def copy_attention_params(pytorch_attn, flax_params):
    """Copy parameters from PyTorch Attention to Flax CausalAttn."""
    print("\nCopying attention parameters...")
    
    # Split PyTorch's combined qkv weights
    qkv_weight = pytorch_attn.w_qkv.weight.detach().numpy()
    q_weight, k_weight, v_weight = np.split(qkv_weight, 3, axis=0)
    print(f"PyTorch parameter shapes before copy:")
    print(f"Query: {q_weight.shape}")
    print(f"Key: {k_weight.shape}")
    print(f"Value: {v_weight.shape}")
    print(f"Output: {pytorch_attn.w_out.weight.shape}")

    # Print parameter shapes for verification
    print("Flax parameter shapes before copy:")
    print(jax.tree.map(lambda x: x.shape, flax_params['params']))

    # Reshape for Flax's dense general format [D, H, Dh]
    def reshape_for_flax(w, n_heads, head_dim):
        return w.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)  # [D, H, Dh]
    
    n_heads = pytorch_attn.n_heads
    head_dim = pytorch_attn.head_dim
    
    new_params = {
        'query': {'kernel': reshape_for_flax(q_weight, n_heads, head_dim)},
        'key': {'kernel': reshape_for_flax(k_weight, n_heads, head_dim)},
        'value': {'kernel': reshape_for_flax(v_weight, n_heads, head_dim)},
        'attn_out_proj': {'kernel': reshape_for_flax(pytorch_attn.w_out.weight.detach().numpy(), n_heads, head_dim).transpose(1, 2, 0)},
    }
    
    # Print parameter shapes for verification
    print("Parameter shapes after copy:")
    for k in new_params:
        print(f"{k}: {new_params[k]['kernel'].shape}")
    
    return {'params': new_params}

def compare_attention_outputs(dim=256, n_heads=4, seq_len=10, batch_size=2):
    """Compare attention outputs between implementations."""
    # Initialize modules
    torch_attn = init_pytorch_attention(dim, n_heads)
    flax_attn = init_flax_attention(dim, n_heads)
    
    # Initialize Flax params with PyTorch weights
    dummy_input = jnp.ones((batch_size, seq_len, dim))
    flax_params = flax_attn.init(jax.random.PRNGKey(0), dummy_input)
    flax_params = copy_attention_params(torch_attn, flax_params)
    
    # Generate random input
    np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    torch_input = torch.tensor(np_input)
    
    # PyTorch forward pass
    with torch.no_grad():
        # Precompute rotary embeddings
        freqs_cis = precompute_freqs_cis(
            dim // n_heads, 
            seq_len,
            theta=500000  # match plainlm_model's Transformer config
        )
        torch_output = torch_attn(torch_input, freqs_cis).numpy()
    
    # Initialize JAX rotary embeddings
    jax_freqs = init_jax_rope(dim, seq_len, n_heads)
    
    # Flax forward pass
    # Split input into Q, K, V
    q, k, v = jnp.split(jnp.array(np_input), 3, axis=-1)
    # Apply rotary embeddings
    q, k = apply_rope_jax(q, k, jax_freqs)
    # Recombine and pass through attention
    flax_output = flax_attn.apply(flax_params, jnp.concatenate([q, k, v], axis=-1))
    flax_output = np.array(flax_output)
    
    # Calculate differences
    mse = np.mean((torch_output - flax_output) ** 2)
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
        'dim': 256,
        'n_heads': 4,
        'seq_len': 10,
        'batch_size': 2,
    }
    
    # Run comparison
    mse, max_diff = compare_attention_outputs(**config)
    
    print("\nFinal Results:")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")

if __name__ == "__main__":
    main()
