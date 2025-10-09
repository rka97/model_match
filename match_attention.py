from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from match_rope import apply_rope_jax, init_jax_rope
from nanodo_model import CausalAttn, DoConfig
from plainlm_model import (
    Attention,
    ModelConfig,
    precompute_freqs_cis,
)


class AttentionManual(torch.nn.Module):
    """Attention module using manual einsum computation instead of SDPA."""
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads

        self.w_qkv = torch.nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.w_out = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, freqs_cis):
        from plainlm_model import apply_rotary_emb_complex_like
        
        bsz, seqlen, d = x.shape  # (bsz, seqlen, d)

        q, k, v = self.w_qkv(x).split(d, dim=2)  # (bsz, seqlen, d)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)

        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)  # (bsz, seqlen, nh, h_dim)

        # Scale queries
        q = q / (self.head_dim ** 0.5)

        # Compute attention scores using einsum
        # q: (bsz, seqlen, nh, h_dim), k: (bsz, seqlen, nh, h_dim)
        # att: (bsz, nh, seqlen, seqlen)
        att = torch.einsum("bqhd,bkhd->bhqk", q, k)

        # Create causal mask
        mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=x.device))
        mask = mask.view(1, 1, seqlen, seqlen)

        # Apply mask and softmax
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Compute attention output
        # att: (bsz, nh, seqlen, seqlen), v: (bsz, seqlen, nh, h_dim)
        # out: (bsz, seqlen, nh, h_dim)
        out = torch.einsum("bhqk,bkhd->bqhd", att, v)

        # Reshape and project output
        out = out.contiguous().view(bsz, seqlen, d)  # (bsz, seqlen, d)

        return self.w_out(out)


def init_pytorch_attention(dim=256, n_heads=4, seq_len=128, compile_model=True, use_manual_attn=False):
    """Initialize PyTorch attention module from plainlm_model."""
    print(f"Initializing PyTorch Attention with dim={dim}, n_heads={n_heads}, compile={compile_model}, manual={use_manual_attn}")
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
    
    # Choose attention implementation
    if use_manual_attn:
        attn = AttentionManual(config)
    else:
        from plainlm_model import Attention
        attn = Attention(config)
    
    # Compile the attention module if requested
    if compile_model:
        print("Compiling PyTorch attention with torch.compile...")
        attn = torch.compile(attn)
    
    return attn, freqs_cis


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

    # Handle both Attention and AttentionManual
    if isinstance(pytorch_attn, torch.nn.Module):
        # Get the underlying module if compiled
        if hasattr(pytorch_attn, '_orig_mod'):
            pytorch_attn = pytorch_attn._orig_mod
    
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


def compare_attention_outputs(dim=256, n_heads=4, seq_len=10, batch_size=2, num_trials=100, compile_pytorch=True, use_manual_attn=False):
    """Compare attention outputs between implementations."""
    # Initialize modules
    torch_attn, freqs_cis = init_pytorch_attention(dim, n_heads, seq_len, compile_model=compile_pytorch, use_manual_attn=use_manual_attn)
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

    # Timing comparison
    print(f"\n{'=' * 50}")
    print(f"Running timing comparison with {num_trials} trials...")
    print(f"PyTorch compiled: {compile_pytorch}")
    print(f"PyTorch using manual attention: {use_manual_attn}")
    print(f"{'=' * 50}")

    torch_times = []
    flax_times = []

    # Warmup runs
    print("Warming up (this may take longer for compiled models)...")
    for _ in range(10):
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        with torch.no_grad():
            _ = torch_attn(torch.tensor(np_input), freqs_cis)
        _ = flax_attn.apply(flax_params, jnp.array(np_input))
    
    # JAX block_until_ready to ensure compilation is complete
    jax.block_until_ready(flax_attn.apply(flax_params, jnp.array(np_input)))

    print(f"Starting {num_trials} timed trials...")
    for i in range(num_trials):
        # Generate new random input for each trial
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        torch_input = torch.tensor(np_input)
        jax_input = jnp.array(np_input)

        # Time PyTorch
        torch_start = time.perf_counter()
        with torch.no_grad():
            _ = torch_attn(torch_input, freqs_cis)
        torch_end = time.perf_counter()
        torch_times.append(torch_end - torch_start)

        # Time JAX/Flax
        flax_start = time.perf_counter()
        flax_result = flax_attn.apply(flax_params, jax_input)
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
    print(f"\nPyTorch Attention (compiled={compile_pytorch}, manual={use_manual_attn}):")
    print(f"  Mean: {np.mean(torch_times):.4f} ms")
    print(f"  Median: {np.median(torch_times):.4f} ms")
    print(f"  Std Dev: {np.std(torch_times):.4f} ms")
    print(f"  Min: {np.min(torch_times):.4f} ms")
    print(f"  Max: {np.max(torch_times):.4f} ms")

    print(f"\nJAX/Flax Attention:")
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


def main():
    print("=" * 50)
    print("Comparing Attention Implementations")
    print("=" * 50)

    # Test configuration
    config = {
        "dim": 1024,
        "n_heads": 4,
        "seq_len": 1024,
        "batch_size": 16,
        "num_trials": 10,  # Number of timing trials
        "compile_pytorch": True,  # Enable torch.compile
        "use_manual_attn": False,  # Use manual einsum-based attention
    }

    # Run comparison
    mse, max_diff, torch_times, flax_times = compare_attention_outputs(**config)

    print(f"\n{'=' * 50}")
    print("Final Summary:")
    print(f"{'=' * 50}")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Maximum Absolute Difference: {max_diff:.8f}")
    print(f"PyTorch Mean Time: {np.mean(torch_times):.4f} ms")
    print(f"JAX Mean Time: {np.mean(flax_times):.4f} ms")


if __name__ == "__main__":
    main()