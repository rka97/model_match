#!/usr/bin/env python3

from typing import NamedTuple, Optional, Union, Any, Callable
import chex
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from optax import tree_utils as otu
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils


@jax.jit
def precondition_grad(
    g: jnp.ndarray, L_prev: Optional[jnp.ndarray]
) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Compute preconditioned gradient and update preconditioner matrix.

    Args:
        g: Gradient matrix of shape (d_L, d_R)
        L_prev: Previous preconditioner matrix of shape (d_L, d_L) or None

    Returns:
        Tuple of (preconditioned gradient, updated preconditioner)
    """
    if g.ndim != 2 or L_prev is None:
        return g, L_prev

    # Compute gradient for the preconditioner: G_t ← ∇L_t(X_{t-1})
    G_t = g
    d_R = G_t.shape[1]

    # Update the preconditioner: L_t ← L_{t-1} + (1/d_R) G_t G_t^T
    G_t_outer = jnp.matmul(G_t, G_t.T) / d_R
    L_t = L_prev + G_t_outer

    # Compute the preconditioned gradient
    # \hat{G}_t ← L_t^{-1/2} G_t
    L_t_neg_half = _inv_sqrtm(L_t)
    return jnp.matmul(L_t_neg_half, G_t), L_t


@jax.jit
def _inv_sqrtm(A: jnp.ndarray) -> jnp.ndarray:
    """Compute inverse matrix square root using SVD.

    Args:
        A: Positive definite matrix to compute inverse sqrt of

    Returns:
        A^{-1/2} computed via SVD decomposition
    """
    U, S, Vh = jnp.linalg.svd(A, hermitian=True)
    sqrt_S_inv = 1.0 / jnp.sqrt(S)
    return U @ jnp.diag(sqrt_S_inv) @ Vh


def double_tree_map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None
) -> Any:
    leaves, treedef = tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    applied = [f(*xs) for xs in zip(*all_leaves)]
    all_leaves_f1 = [a[0] for a in applied]
    all_leaves_f2 = [a[1] for a in applied]
    return treedef.unflatten(all_leaves_f1), treedef.unflatten(all_leaves_f2)


class OneSidedShampooState(NamedTuple):
    """State for the One-sided Shampoo algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    L: base.Updates  # Preconditioner
    mu: base.Updates  # Momentum


def scale_by_one_sided_shampoo(
    learning_rate: base.ScalarOrSchedule,
    beta: float = 0.9,
    epsilon: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """Rescale updates according to the One-sided Shampoo algorithm.

    One-sided Shampoo is a variant of Shampoo that uses only the right preconditioner
    for optimization. It is more memory-efficient than the full Shampoo optimizer
    while still providing adaptive learning rates per feature.

    Args:
      learning_rate: A global scaling factor.
      beta: Exponential decay rate for the momentum estimates.
      epsilon: Small constant added for numerical stability.
      mu_dtype: Optional dtype for the momentum accumulator.

    Returns:
      A `GradientTransformation` object.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        # Initialize the preconditioner as identity matrices
        def init_preconditioner(param):
            if param.ndim == 2:
                d_L, d_R = param.shape
                return epsilon * jnp.eye(d_L, dtype=param.dtype)
            else:
                return None

        # Initialize the momentum to zeros
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        L = jax.tree.map(init_preconditioner, params)

        return OneSidedShampooState(
            count=jnp.zeros([], jnp.int32),
            L=L,
            mu=mu,
        )

    def update_fn(updates, state, params=None):
        del params

        count_inc = numerics.safe_increment(state.count)

        # Update the momentum
        mu = otu.tree_update_moment(updates, state.mu, beta, 1)
        mu_hat = otu.tree_bias_correction(mu, beta, count_inc)

        # Update all preconditioners and compute all preconditioned gradients
        preconditioned_grads, new_L = double_tree_map(
            lambda g, L_prev: precondition_grad(g, L_prev), mu_hat, state.L
        )

        # Scale by learning rate
        final_updates = jax.tree.map(lambda g: -g * learning_rate, preconditioned_grads)

        mu = otu.tree_cast(mu, mu_dtype)

        return final_updates, OneSidedShampooState(
            count=count_inc,
            L=new_L,
            mu=mu,
        )

    return base.GradientTransformation(init_fn, update_fn)


def one_sided_shampoo(
    learning_rate: base.ScalarOrSchedule,
    beta: float = 0.9,
    epsilon: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
) -> base.GradientTransformation:
    """One-sided Shampoo optimizer.

    One-sided Shampoo is a variant of Shampoo that only uses the right preconditioner
    for optimization. This implementation uses the One-sided Shampoo optimizer for
    2D parameters and falls back to AdamW for non-2D parameters.

    Args:
      learning_rate: A global scaling factor.
      beta: Exponential decay rate for the momentum estimates in Shampoo.
      epsilon: Small constant added for numerical stability.
      mu_dtype: Optional dtype for the momentum accumulator.
      adam_b1: Exponential decay rate for Adam's first moment estimates.
      adam_b2: Exponential decay rate for Adam's second moment estimates.
      adam_eps_root: Epsilon to stabilize division in Adam, square root version.
      adam_weight_decay: Weight decay factor for Adam.

    Returns:
      The corresponding `GradientTransformation`.
    """
    return combine.multi_transform(
        transforms={
            "shampoo": scale_by_one_sided_shampoo(
                learning_rate=learning_rate,
                beta=beta,
                epsilon=epsilon,
                mu_dtype=mu_dtype,
            ),
            "adam": alias.adamw(
                learning_rate=learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=epsilon,
                eps_root=adam_eps_root,
                weight_decay=adam_weight_decay,
                mu_dtype=mu_dtype,
            ),
        },
        param_labels=lambda params: jax.tree.map(
            lambda x: "shampoo" if x.ndim == 2 else "adam", params
        ),
    )
