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
    initial_loss = None
    for i in range(100):
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params, X, Y)
        if initial_loss is None:
            initial_loss = loss
        losses.append(loss)
        print(f"Step {i}: loss={loss:.4f}")

        # Early stopping if loss explodes
        if loss > 10 * initial_loss:
            print(f"Early stopping - loss exceeded 10x initial loss")
            break

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


def test_feedforward_network():
    """Test adaptive one-sided shampoo on a simple feedforward network."""
    rng = jax.random.PRNGKey(42)
    np.random.seed(42)

    # Network architecture
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    batch_size = 8

    # Generate synthetic data
    X = np.random.randn(batch_size, input_dim)
    Y = np.random.randn(batch_size, output_dim)

    # Initialize network parameters
    params = {
        'w1': jax.random.normal(rng, (input_dim, hidden_dim)) * 0.01,
        'b1': jnp.zeros(hidden_dim),
        'w2': jax.random.normal(rng, (hidden_dim, output_dim)) * 0.01,
        'b2': jnp.zeros(output_dim)
    }

    # Define network and loss
    def forward(params, x):
        h = jnp.dot(x, params['w1']) + params['b1']
        h = jax.nn.relu(h)
        return jnp.dot(h, params['w2']) + params['b2']

    def loss_fn(params, x, y):
        pred = forward(params, x)
        return 0.5 * jnp.mean((pred - y) ** 2)

    # Initialize optimizer
    optimizer = adaptive_one_sided_shampoo(learning_rate=0.1)
    opt_state = optimizer.init(params)

    # Training loop
    losses = []
    initial_loss = None
    for i in range(100):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, Y)
        if initial_loss is None:
            initial_loss = loss
        losses.append(loss)
        print(f"Step {i}: loss={loss:.4f}")

        # Early stopping if loss explodes
        if loss > 10 * initial_loss:
            print(f"Early stopping - loss exceeded 10x initial loss")
            break

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = apply_updates(params, updates)

    # Verify training worked
    assert losses[-1] < losses[0], "Loss should decrease"
    print(f"Feedforward test - Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    # test_adaptive_one_sided_shampoo()
    test_feedforward_network()
