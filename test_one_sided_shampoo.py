import jax
import jax.numpy as jnp
import numpy as np
from one_sided_shampoo import one_sided_shampoo, _inv_sqrtm, precondition_grad
from optax import apply_updates


def test_precondition_grad():
    """Test that precondition_grad correctly computes preconditioned gradients."""
    rng = jax.random.PRNGKey(42)
    g = jax.random.normal(rng, (5, 10))  # d_L=5, d_R=10
    L_prev = jnp.eye(5)  # Initial preconditioner

    # Compute preconditioned gradient
    preconditioned_g, L_new = precondition_grad(g, L_prev)

    # Verify shapes
    assert preconditioned_g.shape == g.shape
    assert L_new.shape == (5, 5)

    # Verify preconditioner was updated
    assert not jnp.allclose(L_new, L_prev)

    # Verify gradient was transformed
    assert not jnp.allclose(preconditioned_g, g)
    print("precondition_grad test passed")


def test_precondition_grad_standalone():
    """Test standalone precondition_grad function."""
    rng = jax.random.PRNGKey(42)
    g = jax.random.normal(rng, (5, 10))  # d_L=5, d_R=10
    L_prev = jnp.eye(5)  # Initial preconditioner

    # Test with valid 2D input
    preconditioned_g, L_new = precondition_grad(g, L_prev)
    print(preconditioned_g)
    print(L_new)
    assert preconditioned_g.shape == g.shape
    assert L_new.shape == (5, 5)
    assert not jnp.allclose(L_new, L_prev)
    assert not jnp.allclose(preconditioned_g, g)

    # Test with None preconditioner
    preconditioned_g, L_new = precondition_grad(g, None)
    assert preconditioned_g.shape == g.shape
    assert L_new is None
    assert jnp.allclose(preconditioned_g, g)

    print("standalone precondition_grad test passed")


def test_inv_sqrtm():
    """Test that _inv_sqrtm correctly computes matrix inverse square root."""
    # Create a random positive definite matrix
    rng = jax.random.PRNGKey(42)
    A = jax.random.normal(rng, (5, 5))
    A = A @ A.T  # Make symmetric positive definite

    # Compute inverse sqrt
    A_inv_sqrt = _inv_sqrtm(A)

    # Verify A^{-1/2} @ A^{-1/2} ≈ A^{-1}
    reconstructed_inv = A_inv_sqrt @ A_inv_sqrt
    true_inv = jnp.linalg.inv(A)

    error = jnp.linalg.norm(reconstructed_inv - true_inv)
    print(f"Inverse sqrt reconstruction error: {error:.4e}")
    assert error < 1e-3, "A^{-1/2} @ A^{-1/2} should approximate A^{-1}"


def test_one_sided_shampoo():
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
    optimizer = one_sided_shampoo(learning_rate=0.1)
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
    test_inv_sqrtm()
    test_precondition_grad_standalone()
    test_one_sided_shampoo()
