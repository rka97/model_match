import torch
import jax
import jax.numpy as jnp
import numpy as np
from nanodo_model import TransformerDo, DoConfig
from plainlm_model import Transformer, ModelConfig


def init_nanodo_model():
    """Initialize the JAX NanoDO model."""
    print("Initializing NanoDO (JAX) model...")
    
    # Initialize model configuration
    B, L = (1, 128)  # Batch size, sequence length
    cfg = DoConfig(D=128, H=4, L=L, N=2, V=256, F=4 * 128)
    model = TransformerDo(cfg)
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(42)
    dummy_input = jax.random.randint(rng_key, shape=(B, 1), minval=0, maxval=cfg.V, dtype=jnp.int32)
    params = model.init(rng_key, dummy_input)
    
    print(f"NanoDO model initialized with configuration:")
    print(f"  - Model dimension (D): {cfg.D}")
    print(f"  - Number of heads (H): {cfg.H}")
    print(f"  - Max sequence length (L): {cfg.L}")
    print(f"  - Number of layers (N): {cfg.N}")
    print(f"  - Vocabulary size (V): {cfg.V}")
    
    return model, params, cfg


def init_plainlm_model():
    """Initialize the PyTorch PlainLM model."""
    print("Initializing PlainLM (PyTorch) model...")
    
    # Define model configuration to match NanoDO
    config = ModelConfig(
        vocab_size=256,    # Match NanoDO's vocab size
        seq_len=128,       # Match NanoDO's sequence length
        dim=128,           # Match NanoDO's model dimension
        expand=4.0,        # MLP expansion factor
        n_layers=2,        # Match NanoDO's layer count
        n_heads=4,         # Match NanoDO's head count
        mlp='mlp',         # Using standard MLP
        rmsorm_eps=1e-6,   # RMSNorm epsilon
        tie_embeddings=True # Tie embedding and output weights
    )
    
    # Initialize model
    model = Transformer(config)
    model.eval()  # Set to evaluation mode
    
    print(f"PlainLM model initialized with configuration:")
    print(f"  - Model dimension: {config.dim}")
    print(f"  - Number of heads: {config.n_heads}")
    print(f"  - Max sequence length: {config.seq_len}")
    print(f"  - Number of layers: {config.n_layers}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    
    return model, config


def compare_predictions(input_sequence, k=5):
    """Compare predictions from both models."""
    print("\n" + "="*50)
    print(f"Comparing predictions for {k} tokens")
    print("="*50)
    
    # Initialize both models
    nanodo_model, nanodo_params, nanodo_cfg = init_nanodo_model()
    plainlm_model, plainlm_cfg = init_plainlm_model()
    
    # Convert input to appropriate formats
    jax_input = jnp.array(input_sequence, dtype=jnp.int32)
    torch_input = torch.tensor(input_sequence, dtype=torch.long)
    
    print(f"\nInput sequence: {input_sequence}")
    
    # Get predictions from NanoDO (JAX)
    print("\nGenerating predictions with NanoDO (JAX)...")
    _, nanodo_predictions = nanodo_model.apply(nanodo_params, jax_input, k, method=nanodo_model.predict)
    nanodo_predictions = np.array(nanodo_predictions).flatten()
    
    # Get predictions from PlainLM (PyTorch)
    print("Generating predictions with PlainLM (PyTorch)...")
    with torch.no_grad():
        _, plainlm_predictions = plainlm_model.predict(torch_input, k)
    plainlm_predictions = plainlm_predictions.numpy().flatten()
    
    # Compare predictions
    print("\nPredictions:")
    print(f"NanoDO (JAX): {nanodo_predictions}")
    print(f"PlainLM (PyTorch): {plainlm_predictions}")
    
    # Check if predictions match
    match = np.array_equal(nanodo_predictions, plainlm_predictions)
    print(f"\nPredictions match: {match}")
    
    if not match:
        # Calculate how many tokens match
        matching_tokens = sum(n == p for n, p in zip(nanodo_predictions, plainlm_predictions))
        print(f"Number of matching tokens: {matching_tokens}/{k} ({matching_tokens/k*100:.1f}%)")
    
    return nanodo_predictions, plainlm_predictions


def main():
    """Main function to run the comparison."""
    print("Initializing model comparison...")
    
    # Create a simple input sequence
    # Using a batch size of 1 for simplicity
    input_sequence = [[1, 2, 3, 4, 5]]
    
    # Compare predictions for 5 tokens
    nanodo_preds, plainlm_preds = compare_predictions(input_sequence, k=5)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
