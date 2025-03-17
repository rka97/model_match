import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from nanodo_model import TransformerDo, DoConfig
from plainlm_model import Transformer, ModelConfig


def init_nanodo_model():
    """Initialize the JAX NanoDO model."""
    print("Initializing NanoDO (JAX) model...")

    # Initialize model configuration
    B, L = (1, 128)  # Batch size, sequence length
    cfg = DoConfig(
        D=128,                           # Model dimension
        H=4,                             # Number of heads
        L=L,                             # Sequence length
        N=2,                             # Number of layers
        V=256,                           # Vocabulary size
        F=4 * 128,                       # Feed-forward dimension
        rmsnorm_epsilon=1e-6,            # RMSNorm epsilon
        kernel_init=nn.initializers.xavier_uniform(),
        embed_init=nn.initializers.normal(stddev=0.02)
    )
    model = TransformerDo(cfg)

    rng_key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((B, L), dtype=jnp.int32)
    params = model.init(rng_key, dummy_input)

    print(f"NanoDO model initialized with configuration:")
    print(f"  - Model dimension (D): {cfg.D}")
    print(f"  - Number of heads (H): {cfg.H}")
    print(f"  - Max sequence length (L): {cfg.L}")
    print(f"  - Number of layers (N): {cfg.N}")
    print(f"  - Vocabulary size (V): {cfg.V}")
    print(f"  - Initialization: Xavier Uniform + normal(0, 0.02) for embeddings")

    return model, params, cfg


def init_plainlm_model():
    """Initialize the PyTorch PlainLM model."""
    print("Initializing PlainLM (PyTorch) model...")

    torch.manual_seed(42)
    
    # Define model configuration to match NanoDO
    config = ModelConfig(
        vocab_size=256,    # Match NanoDO's vocab size
        seq_len=128,       # Match NanoDO's sequence length
        dim=128,           # Match NanoDO's model dimension
        expand=4.0,        # MLP expansion factor (4 * dimension)
        n_layers=2,        # Match NanoDO's layer count
        n_heads=4,         # Match NanoDO's head count
        rmsnorm_eps=1e-6,  # RMSNorm epsilon
        tie_embeddings=True # Tie embedding and output weights to match JAX
    )

    # Initialize model
    model = Transformer(config)
    model.eval()  # Set to evaluation mode
    
    # Manually verify initialization to match JAX
    for name, param in model.named_parameters():
        if 'embed_tokens' in name:
            print(f"  Embedding init: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        if 'lm_head' in name and not model.cfg.tie_embeddings:
            print(f"  LM head init: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

    print(f"PlainLM model initialized with configuration:")
    print(f"  - Model dimension: {config.dim}")
    print(f"  - Number of heads: {config.n_heads}")
    print(f"  - Max sequence length: {config.seq_len}")
    print(f"  - Number of layers: {config.n_layers}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Initialization: Xavier Uniform + normal(0, 0.02) for embeddings")

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

    # Step-by-step debugging
    print("\n---- Step-by-step prediction comparison ----")
    extended_jax_input = jax_input
    extended_torch_input = torch_input.clone()
    
    for step in range(k):
        print(f"\nStep {step+1}/{k}:")
        
        # NanoDO (JAX) - Get logits for the current sequence
        jax_logits = nanodo_model.apply(nanodo_params, extended_jax_input)
        # Get the logits for the last token
        jax_next_token_logits = jax_logits[:, -1, :]
        # Get top 5 likely tokens
        jax_top5_indices = jnp.argsort(-jax_next_token_logits[0])[:5]
        jax_top5_probs = jax.nn.softmax(jax_next_token_logits[0])[jax_top5_indices]
        jax_next_token = jnp.argmax(jax_next_token_logits, axis=-1)
        
        with torch.no_grad():
            torch_logits = plainlm_model(extended_torch_input)
            # Get the logits for the last token
            torch_next_token_logits = torch_logits[:, -1, :]
            # Get top 5 likely tokens
            torch_top5_indices = torch.topk(torch_next_token_logits[0], 5)[1]
            torch_top5_probs = torch.softmax(torch_next_token_logits[0], dim=-1)[torch_top5_indices]
            torch_next_token = torch.argmax(torch_next_token_logits, dim=-1)
        
        # Print token predictions and probabilities
        print(f"  JAX next token: {jax_next_token[0]}, logit: {jax_next_token_logits[0, jax_next_token[0]]:.2f}")
        print(f"  PyTorch next token: {torch_next_token.item()}, logit: {torch_next_token_logits[0, torch_next_token].item():.2f}")
        
        print("  JAX top 5 tokens and probabilities:")
        for i, (idx, prob) in enumerate(zip(jax_top5_indices, jax_top5_probs)):
            print(f"    {i+1}. Token {idx} (prob: {prob:.6f})")
            
        print("  PyTorch top 5 tokens and probabilities:")
        for i, (idx, prob) in enumerate(zip(torch_top5_indices.numpy(), torch_top5_probs.numpy())):
            print(f"    {i+1}. Token {idx} (prob: {prob:.6f})")
        
        # Extend inputs with predicted tokens
        extended_jax_input = jnp.concatenate([extended_jax_input, jax_next_token[:, None]], axis=1)
        extended_torch_input = torch.cat([extended_torch_input, torch_next_token.unsqueeze(1)], dim=1)
    
    # Get full predictions 
    print("\n---- Full prediction results ----")
    
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


def simple_model_test():
    """
    Test a simplified model to ensure JAX and PyTorch can match exactly.
    Using simple weight matrices with the same values in both frameworks.
    """
    print("\n" + "="*50)
    print("SIMPLE MODEL TEST: JAX and PyTorch with identical weights")
    print("="*50)
    
    vocab_size = 100
    embedding_dim = 16
    
    # Create identical embedding matrix
    np_embedding = np.random.RandomState(42).normal(0, 0.02, (vocab_size, embedding_dim))
    
    # JAX model setup
    jax_embed = jnp.array(np_embedding)
    
    # PyTorch model setup
    torch_embed = torch.nn.Embedding(vocab_size, embedding_dim)
    with torch.no_grad():
        torch_embed.weight.copy_(torch.tensor(np_embedding))
    
    # Create a simple input sequence
    input_ids = [1, 2, 3, 4, 5]
    jax_input = jnp.array([input_ids], dtype=jnp.int32)
    torch_input = torch.tensor([input_ids], dtype=torch.long)
    
    # Get embeddings
    jax_embeddings = jnp.take(jax_embed, jax_input, axis=0)
    torch_embeddings = torch_embed(torch_input)
    
    # Convert to numpy for comparison
    jax_result = np.array(jax_embeddings).reshape(-1)
    torch_result = torch_embeddings.detach().numpy().reshape(-1)
    
    # Compare the embeddings
    print(f"JAX embeddings: shape={jax_embeddings.shape}")
    print(f"PyTorch embeddings: shape={torch_embeddings.shape}")
    
    # Check if they match
    # Allow for small numeric differences due to floating point
    match = np.allclose(jax_result, torch_result, rtol=1e-5, atol=1e-5)
    print(f"Embeddings match: {match}")
    
    if not match:
        print(f"Max absolute difference: {np.max(np.abs(jax_result - torch_result))}")
    
    return match

def main():
    """Main function to run the comparison."""
    print("Initializing model comparison...")
    
    # First, test a simple model with identical weights
    simple_match = simple_model_test()
    
    # Create a simple input sequence
    # Using a batch size of 1 for simplicity
    input_sequence = [[1, 2, 3, 4, 5]]

    nanodo_preds, plainlm_preds = compare_predictions(input_sequence, k=5)

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
