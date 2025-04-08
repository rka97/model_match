import jax
import jax.numpy as jnp
import numpy as np
from nanodo_model import DoConfig, TransformerDo
from pathlib import Path

def data_generator(
    rng_key: jax.random.PRNGKey, batch_size: int, seq_len: int, V: int = 16
) -> jnp.ndarray:
    """Generate sequences where:
    - t[i][0] is random in [0,V_prime-1]
    - t[i][1] is random in [0,V_prime-1]
    - t[i][2] is random in [V_prime,V-1] (operation selector)
    - t[i][j+3] follows pattern based on op:
        op=V_prime: (t0 + j*t1) % V_prime
        op=V_prime+1: (t0 - j*t1) % V_prime
        op=V_prime+2: (t0 * (t1**j)) % V_prime

    Args:
        rng_key: JAX random key
        batch_size: Number of sequences to generate
        seq_len: Length of each sequence
        V: Vocabulary size (must be >=4 to have at least 3 operations)

    Returns:
        Tensor of shape (batch_size, seq_len) with values in 0-V-1
    """
    assert V >= 4, "Vocabulary size must be >=4 to support operations"
    V_prime = V - 3

    rng1, rng2, rng3 = jax.random.split(rng_key, 3)

    # Generate random starting values (t0), multipliers (t1), and operations (op)
    t0 = jax.random.randint(rng1, (batch_size,), 0, V_prime)
    t1 = jax.random.randint(rng2, (batch_size,), 0, V_prime)
    op = jax.random.randint(rng3, (batch_size,), V_prime, V)

    # Initialize sequences with t0, t1 and op
    sequences = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    sequences = sequences.at[:, 0].set(t0)
    sequences = sequences.at[:, 1].set(t1)
    sequences = sequences.at[:, 2].set(op)

    # Compute remaining elements based on operation
    for j in range(seq_len - 3):
        j_val = j + 1  # Start from j=1 for the pattern

        # Calculate all three possible operations
        add = (t0 + j_val * t1) % V_prime
        sub = (t0 - j_val * t1) % V_prime
        mul = (t0 * (t1**j_val)) % V_prime

        # Select based on operation
        result = jnp.select(
            condlist=[op == V_prime, op == V_prime + 1, op == V_prime + 2],
            choicelist=[add, sub, mul],
        )
        sequences = sequences.at[:, j + 3].set(result)

    return sequences


def test_data_generator():
    """Test the data generator and model on generated sequences."""
    # Initialize RNG
    rng_key = jax.random.PRNGKey(42)

    # Generate some test data
    batch_size = 10
    seq_len = 16
    V = 16  # Test with larger vocabulary
    data = data_generator(rng_key, batch_size, seq_len, V)

    print("Generated sequences:")
    for i in range(batch_size):
        print(f"Sequence {i}: {data[i]}")
        # Verify the pattern
        t0, t1, op = data[i, 0], data[i, 1], data[i, 2]
        V_prime = V - 3

        for j in range(seq_len - 3):
            j_val = j + 1
            if op == V_prime:  # Addition
                expected = (t0 + j_val * t1) % V_prime
            elif op == V_prime + 1:  # Subtraction
                expected = (t0 - j_val * t1) % V_prime
            elif op == V_prime + 2:  # Multiplication
                expected = (t0 * (t1**j_val)) % V_prime
            else:
                expected = -1  # Invalid operation

            actual = data[i, j + 3]
            if actual != expected:
                print(f"  Error at pos {j+3}: expected {expected}, got {actual}")

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
