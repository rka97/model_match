import jax
import jax.numpy as jnp
import numpy as np
from adaptive_one_sided_shampoo import adaptive_one_sided_shampoo
from optax import apply_updates

def test_adaptive_one_sided_shampoo():
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(42)
    np.random.seed(42)

    # Generate synthetic data: y = W_true @ x + noise
    n_samples = 2
    n_features = 5
    output_dim = 10
    W_true = np.random.randn(n_features, output_dim)
    X = np.random.randn(n_samples, n_features)
    Y = X @ W_true + 0.1 * np.random.randn(n_samples, output_dim)

    # Initialize parameters (weights)
    params = {"W": jnp.zeros((n_features, output_dim))}

    # Define loss function: MSE
    def loss_fn(params, X, Y):
        pred = jnp.dot(X, params["W"])
        return 0.5 * jnp.mean((pred - Y) ** 2)

    # Initialize optimizer
    optimizer = adaptive_one_sided_shampoo(learning_rate=0.1)
    opt_state = optimizer.init(params)

    # Training loop
    losses = []
    for _ in range(100):
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params, X, Y)
        losses.append(loss)
        print(f"loss={loss}")

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = apply_updates(params, updates)

    # Verify loss decreases
    assert losses[-1] < losses[0], "Loss should decrease during training"
    print(f"Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")

    # Verify learned weights are close to true weights
    true_final_loss = 0.5 * jnp.mean((jnp.dot(X, W_true) - Y) ** 2)
    error = losses[-1] - true_final_loss
    print(f"Final loss error: {error:.6f}")
    assert error < 0.5, "Learned weights should be close to true weights"


if __name__ == "__main__":
    test_adaptive_one_sided_shampoo()
