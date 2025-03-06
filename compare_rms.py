import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from plainlm_model import RMSNorm as CustomRMSNorm


def compare_implementations(batch_size=2, seq_len=10, dim=256):
    """Compare the custom RMSNorm with PyTorch's implementation."""
    print(f"Comparing RMSNorm implementations with input shape: ({batch_size}, {seq_len}, {dim})")
    
    # Create identical random inputs
    np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    torch_input = torch.tensor(np_input)
    
    # Initialize both implementations with the same weights
    custom_norm = CustomRMSNorm(dim)
    pytorch_norm = nn.RMSNorm(dim, eps=1e-6)
    
    # Forward pass
    with torch.no_grad():
        custom_output = custom_norm(torch_input)
        pytorch_output = pytorch_norm(torch_input)
    
    # Convert outputs to numpy for comparison
    custom_np = custom_output.numpy()
    pytorch_np = pytorch_output.numpy()
    
    # Calculate differences
    abs_diff = np.abs(custom_np - pytorch_np)
    mean_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    
    # Print results
    print("\nOutput statistics:")
    print(f"Custom  - Mean: {custom_np.mean():.6f}, Std: {custom_np.std():.6f}")
    print(f"PyTorch - Mean: {pytorch_np.mean():.6f}, Std: {pytorch_np.std():.6f}")
    
    print("\nDifference statistics:")
    print(f"Mean absolute difference: {mean_diff:.8f}")
    print(f"Maximum absolute difference: {max_diff:.8f}")
    
    # Check if implementations are effectively equivalent
    is_equivalent = mean_diff < 1e-6
    print(f"\nImplementations are {'equivalent' if is_equivalent else 'different'}")
    
    # Analyze the differences in more detail
    if not is_equivalent:
        print("\nAnalyzing differences in detail:")
        
        # Check normalization calculation
        custom_norm_factor = torch.rsqrt(torch_input.pow(2).mean(-1, keepdim=True) + custom_norm.eps)
        
        # For PyTorch RMSNorm, we need to analyze how it's implemented internally
        # This is an approximation based on the formula
        pytorch_norm_factor = 1.0 / torch.sqrt(torch_input.pow(2).mean(-1, keepdim=True) + pytorch_norm.eps)
        
        norm_factor_diff = torch.abs(custom_norm_factor - pytorch_norm_factor).mean().item()
        print(f"Normalization factor difference: {norm_factor_diff:.8f}")
        
        # Check if the difference is in the normalization or the scaling
        normalized_custom = torch_input * custom_norm_factor
        # Use F.rms_norm directly to get the normalized values without scaling
        normalized_pytorch = F.rms_norm(torch_input, pytorch_norm.normalized_shape, None, pytorch_norm.eps)
        
        norm_diff = torch.abs(normalized_custom - normalized_pytorch).mean().item()
        print(f"Normalized values difference: {norm_diff:.8f}")
    
    return custom_np, pytorch_np, mean_diff, max_diff


def main():
    print("=" * 50)
    print("Comparing RMSNorm Implementations")
    print("=" * 50)
    
    # Compare with default parameters
    custom_output, pytorch_output, mean_diff, max_diff = compare_implementations()
    
    # Compare with different dimensions
    print("\n" + "=" * 50)
    print("Comparing with different dimensions")
    print("=" * 50)
    
    dimensions = [32, 128, 512, 1024]
    for dim in dimensions:
        print(f"\nTesting with dimension: {dim}")
        compare_implementations(dim=dim)


if __name__ == "__main__":
    main()
