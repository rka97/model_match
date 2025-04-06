import jax
import jax.numpy as jnp
from nanodo_model import DoConfig, TransformerDo

def data_generator(rng_key: jax.random.PRNGKey, batch_size: int, seq_len: int, V: int = 16) -> jnp.ndarray:
    """Generate sequences where:
    - t[i][0] is random in [0,V-1]
    - t[i][1] is random in [0,V-1]
    - t[i][j+2] = (t[i][0] + j*t[i][1]) % V for j >= 0
    
    Args:
        rng_key: JAX random key
        batch_size: Number of sequences to generate
        seq_len: Length of each sequence
        
    Returns:
        Tensor of shape (batch_size, seq_len) with values in 0-9
    """
    rng1, rng2 = jax.random.split(rng_key)
    
    # Generate random starting values (t0) and multipliers (t1)
    t0 = jax.random.randint(rng1, (batch_size,), 0, V)
    t1 = jax.random.randint(rng2, (batch_size,), 0, V)
    
    # Initialize sequences with t0 and t1
    sequences = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    sequences = sequences.at[:, 0].set(t0)
    sequences = sequences.at[:, 1].set(t1)
    
    # Compute remaining elements using the formula
    for j in range(seq_len - 2):
        sequences = sequences.at[:, j+2].set((t0 + (j+1)*t1) % V)
    
    return sequences

def test_data_generator():
    """Test the data generator and model on generated sequences."""
    # Initialize RNG
    rng_key = jax.random.PRNGKey(42)
    
    # Generate some test data
    batch_size = 4
    seq_len = 16
    data = data_generator(rng_key, batch_size, seq_len)
    
    print("Generated sequences:")
    for i in range(batch_size):
        print(f"Sequence {i}: {data[i]}")
        # Verify the pattern
        t0, t1 = data[i, 0], data[i, 1]
        for j in range(seq_len - 2):
            expected = (t0 + (j+1)*t1) % 10
            actual = data[i, j+2]
            if actual != expected:
                print(f"  Error at pos {j+2}: expected {expected}, got {actual}")
    
    # Initialize a small model
    config = DoConfig(D=32, H=4, L=seq_len, N=2, V=10, F=32)
    model = TransformerDo(config)
    
    # Initialize model parameters
    init_rng, _ = jax.random.split(rng_key)
    params = model.init(init_rng, data)
    
    # Run forward pass
    logits = model.apply(params, data)
    predictions = jnp.argmax(logits, axis=-1)
    
    print("\nModel predictions:")
    for i in range(batch_size):
        print(f"Sequence {i} input: {data[i]}")
        print(f"Sequence {i} preds: {predictions[i]}")
        print("---")

if __name__ == "__main__":
    test_data_generator()
