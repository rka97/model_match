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
from one_sided_shampoo import precondition_grad, double_tree_map

class AdaptiveOneSidedShampooState(NamedTuple):
    """State for the One-sided Shampoo algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    L: base.Updates  # Preconditioner
    mu: base.Updates  # Momentum
    x0: base.Updates  # Initial state
    grad_sum_sq: base.Updates  # Cumulative sum of squared gradient norms
    r_t: base.Updates  # Track max distance from initial params


def scale_by_adaptive_one_sided_shampoo(
    learning_rate: base.ScalarOrSchedule,
    beta: float = 0.9,
    epsilon: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    R_EPS: float = 1e-4,
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
        x0 = otu.tree_zeros_like(params, dtype=mu_dtype)
        grad_sum_sq = jax.tree.map(lambda x: R_EPS * 1, params)
        r_t = jax.tree.map(lambda x: R_EPS * 1, params)

        return AdaptiveOneSidedShampooState(
            count=jnp.zeros([], jnp.int32),
            L=L,
            mu=mu,
            x0=x0,
            grad_sum_sq=grad_sum_sq,
            r_t=r_t
        )

    def update_fn(updates, state, params):
        count_inc = numerics.safe_increment(state.count)

        # Update the momentum
        mu = otu.tree_update_moment(updates, state.mu, beta, 1)
        mu_hat = otu.tree_bias_correction(mu, beta, count_inc)

        # Update all preconditioners and compute all preconditioned gradients
        preconditioned_grads, new_L = double_tree_map(
            lambda g, L_prev: precondition_grad(g, L_prev), mu_hat, state.L
        )

        current_dist = jax.tree.map(lambda x, y: jnp.linalg.norm(x-y, ord=2), state.x0, params)

        # Compute r_t as max(R_EPS, current_dist, previous r_t)
        r_t = jax.tree.map(
            lambda curr, prev: jnp.maximum(curr, prev),
            current_dist,
            state.r_t
        )

        lr = jax.tree_map(
            lambda r, g: r  * jnp.sqrt(2/g.shape[1]),
            r_t,
            preconditioned_grads
        )
        print(f"lr={lr}")
        # input()
        # Scale by learning rate
        final_updates = jax.tree.map(lambda g, z: -g * z, preconditioned_grads, lr)

        mu = otu.tree_cast(mu, mu_dtype)

        return final_updates, AdaptiveOneSidedShampooState(
            count=count_inc,
            L=new_L,
            mu=mu,
            x0=state.x0,
            grad_sum_sq=state.grad_sum_sq,
            r_t=r_t
        )

    return base.GradientTransformation(init_fn, update_fn)


def adaptive_one_sided_shampoo(
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
            "shampoo": scale_by_adaptive_one_sided_shampoo(
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
