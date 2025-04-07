#!/usr/bin/env python3

# Standard library imports
import os
import sys
import time
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, List, Callable, Any, Dict, Union

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# JAX imports
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn_flax
from flax.training import train_state

# Import from training framework
from abstract_training_pipeline import (
    AbstractModel,
    train,
    Hyperparameters,
    distributed_data_generator,
)

# Suppress JAX dtype warnings about int64 to int32 conversion
warnings.filterwarnings(
    "ignore", message="Explicitly requested dtype.*truncated to dtype int32"
)

# ----- MODEL DEFINITIONS -----


@dataclass
class ModelConfig:
    """Shared configuration for both PyTorch and JAX models."""

    vocab_size: int = 50257
    hidden_dim: int = 256
    embedding_dim: int = 384


class SimpleLanguageModel(AbstractModel):
    """
    A basic language model with just two linear layers (PyTorch implementation).
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        # Use provided config or default
        if config is None:
            config = ModelConfig()

        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Two linear layers
        self.linear1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_seq, target_seq):
        # Get embeddings
        x = self.embedding(input_seq)

        # First linear layer with ReLU activation
        x = F.relu(self.linear1(x))

        # Output projection
        logits = self.linear2(x)

        # Calculate loss
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_seq.view(-1))

        return loss


# JAX model implementation that matches PyTorch model architecture
class SimpleLanguageModelFlax(nn_flax.Module):
    """
    A basic language model with just two linear layers (JAX/Flax implementation).
    """

    vocab_size: int = 50257
    hidden_dim: int = 256
    embedding_dim: int = 384

    @classmethod
    def from_config(cls, config: ModelConfig):
        """Create a model instance from a config object."""
        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
        )

    @nn_flax.compact
    def __call__(self, input_seq, target_seq=None):
        # Get embeddings
        x = nn_flax.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            embedding_init=nn_flax.initializers.normal(stddev=0.02),
            name="embedding",
        )(input_seq)

        # First linear layer with ReLU activation
        x = nn_flax.Dense(
            features=self.hidden_dim,
            kernel_init=nn_flax.initializers.normal(stddev=0.02),
            bias_init=nn_flax.initializers.zeros,
            name="linear1",
        )(x)
        x = jax.nn.relu(x)

        # Output projection
        logits = nn_flax.Dense(
            features=self.vocab_size,
            kernel_init=nn_flax.initializers.normal(stddev=0.02),
            bias_init=nn_flax.initializers.zeros,
            name="linear2",
        )(x)

        # If target_seq is provided, calculate loss (for training)
        if target_seq is not None:
            # Reshape logits and targets for loss calculation
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = target_seq.reshape(-1)

            # Cross-entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits_flat, targets_flat
            ).mean()
            return loss

        # Otherwise just return logits (for inference)
        return logits


# ----- JAX TRAINING UTILITIES -----


def create_train_state(
    rng: jax.Array, config: ModelConfig, batch_size: int, seq_len: int, lr: float = 1e-3
) -> train_state.TrainState:
    """
    Create initial JAX training state with model and optimizer.

    Args:
        rng: JAX random number generator key
        config: Model configuration
        batch_size: Batch size for dummy input
        seq_len: Sequence length for dummy input
        lr: Learning rate for optimizer

    Returns:
        JAX TrainState object with initialized model and optimizer
    """
    # Create a dummy input for model initialization
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Instantiate the model
    model = SimpleLanguageModelFlax.from_config(config)

    # Initialize parameters
    params = model.init(rng, dummy_input, dummy_target)

    # Create optimizer
    tx = optax.adamw(learning_rate=lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)

    # Create and return training state
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnums=(2,))
def train_step(
    state: train_state.TrainState,
    batch: Tuple[jax.Array, jax.Array],
    model_fn: Callable,
) -> Tuple[train_state.TrainState, jax.Array]:
    """
    Perform a single training step with JAX.

    Args:
        state: Current training state
        batch: Tuple of (inputs, targets)
        model_fn: The model's apply function

    Returns:
        Tuple of (new_state, loss)
    """
    inputs, targets = batch

    def loss_fn(params):
        loss = model_fn(params, inputs, targets)
        return loss

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Update parameters
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss


@partial(jax.jit, static_argnums=(2,))
def eval_step(
    state: train_state.TrainState,
    batch: Tuple[jax.Array, jax.Array],
    model_fn: Callable,
) -> jax.Array:
    """
    Perform a single evaluation step with JAX.
    Uses JIT compilation for acceleration.

    Args:
        state: Current training state
        batch: Tuple of (inputs, targets)
        model_fn: The model's apply function

    Returns:
        Loss value
    """
    inputs, targets = batch
    loss = model_fn(state.params, inputs, targets)
    return loss


def jax_data_to_batch(
    inputs: jax.Array, targets: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Prepare JAX data for batch processing.

    The dataloader already provides data in the correct format (batch_size, ...),
    so we just need to make sure it's a jax.Array and return the tuple.

    Args:
        inputs: Input token IDs
        targets: Target token IDs

    Returns:
        Tuple of (inputs, targets) as JAX arrays
    """
    # Ensure we have jax Arrays
    if not isinstance(inputs, jax.Array):
        inputs = jnp.array(inputs)
    if not isinstance(targets, jax.Array):
        targets = jnp.array(targets)

    return (inputs, targets)


def evaluate_jax_model(
    state: train_state.TrainState,
    val_loader: Any,
    model_apply: Callable,
    batch_size: int,
    val_tokens: int,
) -> float:
    """
    Evaluate JAX model on validation data.

    Args:
        state: Current JAX training state
        val_loader: Validation data generator
        model_apply: Model's apply function
        batch_size: Batch size for validation
        val_tokens: Total validation tokens to use

    Returns:
        Average validation loss
    """
    # Compute validation loss
    val_loss = 0.0
    val_steps = val_tokens // batch_size
    for _ in range(val_steps):
        val_inputs, val_targets = next(val_loader)
        val_batch = jax_data_to_batch(val_inputs, val_targets)
        val_loss += eval_step(state, val_batch, model_apply)

    return val_loss / val_steps


# ----- MAIN TRAINING FUNCTIONS -----


def get_default_hyperparameters() -> Hyperparameters:
    """Create default hyperparameters for training."""
    args = Hyperparameters()
    args.train_files = "data/finewebedu_train_*.bin"
    args.val_files = "data/finewebedu_val_*.bin"
    args.train_seq_len = 32  # Smaller sequence length for this simple model
    args.val_seq_len = 32  # Smaller sequence length for validation
    args.val_tokens = 32 * 32  # Number of tokens to validate on
    args.num_iterations = 500  # Fewer iterations for testing
    args.val_loss_every = 50  # Validate every 50 steps
    return args


def train_jax_model(
    args: Optional[Hyperparameters] = None, warmup_steps: int = 5, batch_size: int = 8
) -> float:
    """
    Training function for the JAX model.

    Args:
        args: Training hyperparameters
        warmup_steps: Number of warmup steps to perform (not counted in timing)
                      to account for JIT compilation overhead
        batch_size: Batch size for training

    Returns:
        Total training time in seconds
    """
    if args is None:
        args = get_default_hyperparameters()

    # Model configuration
    config = ModelConfig()

    # Create training state
    rng = jax.random.PRNGKey(42)
    state = create_train_state(
        rng=rng, config=config, batch_size=batch_size, seq_len=args.train_seq_len
    )

    # Create data loaders once and reuse them
    # The dataloader already handles JAX device sharding when framework="jax" is specified
    train_loader = distributed_data_generator(
        args.train_files, batch_size, rank=0, world_size=1, framework="jax"
    )

    # Create validation loader once (will be reused for all validation steps)
    val_loader = distributed_data_generator(
        args.val_files, batch_size, rank=0, world_size=1, framework="jax"
    )

    # Pre-create model instance outside of JIT
    model_apply = SimpleLanguageModelFlax.from_config(config).apply

    # Warmup phase to JIT-compile the functions (not timed)
    print(f"[JAX] Running {warmup_steps} warmup steps for JIT compilation...")
    for _ in range(warmup_steps):
        inputs, targets = next(train_loader)
        batch = jax_data_to_batch(inputs, targets)
        state, _ = train_step(state, batch, model_apply)

    # Warmup the validation function
    # Run once to compile the validation code
    _ = evaluate_jax_model(state, val_loader, model_apply, batch_size, args.val_tokens)

    print("[JAX] Warmup completed, starting timed training...")

    # Training loop (timed)
    total_time = 0

    # Run one step before starting the timer to warm up any remaining compilation
    # This prevents the first step from being artificially slow
    inputs, targets = next(train_loader)
    batch = jax_data_to_batch(inputs, targets)
    state, _ = train_step(state, batch, model_apply)

    # Now start the timer for the actual timed steps
    start_time = time.time()

    for step in range(args.num_iterations + 1):
        # Get batch
        inputs, targets = next(train_loader)
        batch = jax_data_to_batch(inputs, targets)

        # Training step
        state, loss = train_step(state, batch, model_apply)

        # Validation
        if step % args.val_loss_every == 0 or step == args.num_iterations:
            # Measure time for training steps
            end_time = time.time()
            total_time += end_time - start_time

            # Run validation using the extracted function
            val_loss = evaluate_jax_model(
                state, val_loader, model_apply, batch_size, args.val_tokens
            )

            print(
                f"[JAX] Step: {step}/{args.num_iterations}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Training time: {total_time*1000:.0f}ms, "
                f"Ms/step: {total_time*1000/max(step, 1):.2f}"
            )

            # Reset timer for next training segment
            start_time = time.time()

    return total_time


class AdamWWrapper:
    """
    Wrapper for PyTorch's AdamW optimizer that follows the AbstractOptimizer interface.
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        # Group parameters by size for efficiency
        param_groups = []
        for p in params:
            # Only include parameters that require gradients
            if p.requires_grad:
                param_groups.append(p)

        self.optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.param_groups = self.optimizer.param_groups

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def evaluate_pytorch_model(
    model: nn.Module, val_loader: Any, batch_size: int, val_tokens: int
) -> float:
    """
    Evaluate PyTorch model on validation data.

    Args:
        model: PyTorch model to evaluate
        val_loader: Validation data generator
        batch_size: Batch size for validation
        val_tokens: Total validation tokens to use

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0
    val_steps = val_tokens // batch_size

    with torch.no_grad():
        for _ in range(val_steps):
            val_inputs, val_targets = next(val_loader)
            val_loss += model(val_inputs, val_targets).item()

    # Set model back to training mode
    model.train()

    return val_loss / val_steps


def train_pytorch_model(
    args: Optional[Hyperparameters] = None,
    use_compile: bool = True,
    warmup_steps: int = 5,
    batch_size: int = 8,
) -> Optional[float]:
    """
    Training function that uses PyTorch's AdamW optimizer.

    Args:
        args: Training hyperparameters
        use_compile: Whether to use torch.compile for optimization
        warmup_steps: Number of warmup steps for compilation
        batch_size: Batch size for training

    Returns:
        Total training time in seconds, or None if using the abstract_training_pipeline
    """
    if args is None:
        args = get_default_hyperparameters()

    # Enable float32 matmul precision to use TensorFloat32 cores
    torch.set_float32_matmul_precision("high")

    # Custom training logic if we want to use compile and perform warmup
    if use_compile and hasattr(torch, "compile"):
        # Initialize model with config
        config = ModelConfig()
        model = SimpleLanguageModel(config).to("cuda")

        # Compile the model
        print("[PyTorch] Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )

        # Create data loaders once and reuse them
        train_loader = distributed_data_generator(
            args.train_files, batch_size, rank=0, world_size=1, framework="torch"
        )

        # Create validation loader once (will be reused for all validation steps)
        val_loader = distributed_data_generator(
            args.val_files, batch_size, rank=0, world_size=1, framework="torch"
        )

        # Warmup phase (if needed)
        if warmup_steps > 0:
            print(f"[PyTorch] Running {warmup_steps} warmup steps for compilation...")
            model.train()
            for _ in range(warmup_steps):
                inputs, targets = next(train_loader)
                optimizer.zero_grad()
                loss = model(inputs, targets)
                loss.backward()
                optimizer.step()

            # Warmup validation function
            _ = evaluate_pytorch_model(model, val_loader, batch_size, args.val_tokens)

            print("[PyTorch] Warmup completed, starting timed training...")

        # Reset the model state
        model = SimpleLanguageModel(config).to("cuda")
        if use_compile:
            model = torch.compile(model, mode="reduce-overhead")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )

        # Training loop with manual timing
        model.train()
        total_time = 0

        # Run one step before starting the timer to warm up any remaining compilation
        # This prevents the first step from being artificially slow
        inputs, targets = next(train_loader)
        optimizer.zero_grad()
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

        # Now start the timer for the actual timed steps
        start_time = time.time()

        for step in range(args.num_iterations + 1):
            # Get batch
            inputs, targets = next(train_loader)

            # Training step
            optimizer.zero_grad()
            loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

            # Validation
            if step % args.val_loss_every == 0 or step == args.num_iterations:
                # Measure time for training steps
                end_time = time.time()
                total_time += end_time - start_time

                # Run validation using the extracted function
                val_loss = evaluate_pytorch_model(
                    model, val_loader, batch_size, args.val_tokens
                )

                print(
                    f"[PyTorch] Step: {step}/{args.num_iterations}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Training time: {total_time*1000:.0f}ms, "
                    f"Ms/step: {total_time*1000/max(step, 1):.2f}"
                )

                # Reset timer for next training segment
                start_time = time.time()

        return total_time
    else:
        # Use the existing training framework from abstract_training_pipeline
        train(SimpleLanguageModel, [AdamWWrapper], args)
        return None  # No timing information available


# ----- BENCHMARKING AND ENTRY POINT -----


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    num_iterations: int = 500
    val_loss_every: int = 100
    warmup_steps: int = 10
    batch_size: int = 8
    num_runs: int = 2
    use_compile: bool = True


def compare_training(config: Optional[BenchmarkConfig] = None) -> None:
    """
    Compare training speeds of PyTorch and JAX implementations.

    Args:
        config: Optional benchmark configuration
    """
    if config is None:
        config = BenchmarkConfig()

    print("\n" + "=" * 60)
    print("TRAINING SPEED COMPARISON: PyTorch vs JAX")
    print("=" * 60)

    # Set up identical hyperparameters for both frameworks
    args = get_default_hyperparameters()
    args.num_iterations = config.num_iterations
    args.val_loss_every = config.val_loss_every

    pytorch_times = []
    jax_times = []

    for run in range(config.num_runs):
        print(f"\nRun {run+1}/{config.num_runs}")

        # Train with JAX
        print("\nStarting JAX training...")
        jax_time = train_jax_model(
            args=args, warmup_steps=config.warmup_steps, batch_size=config.batch_size
        )
        jax_times.append(jax_time)

        # Train with PyTorch
        print("\nStarting PyTorch training...")
        pytorch_time = train_pytorch_model(
            args=args,
            use_compile=config.use_compile,
            warmup_steps=config.warmup_steps,
            batch_size=config.batch_size,
        )
        if pytorch_time is not None:
            pytorch_times.append(pytorch_time)

    # Calculate averages
    avg_jax_time = sum(jax_times) / len(jax_times)

    # Compare results
    print("\n" + "=" * 60)
    print("TRAINING SPEED RESULTS")
    print("=" * 60)
    print(f"JAX average total training time: {avg_jax_time*1000:.0f}ms")
    print(f"JAX average ms/step: {avg_jax_time*1000/args.num_iterations:.2f}")

    if pytorch_times:
        avg_pytorch_time = sum(pytorch_times) / len(pytorch_times)
        print(f"PyTorch average total training time: {avg_pytorch_time*1000:.0f}ms")
        print(
            f"PyTorch average ms/step: {avg_pytorch_time*1000/args.num_iterations:.2f}"
        )

        # Show comparison
        speedup = avg_pytorch_time / avg_jax_time if avg_jax_time > 0 else float("inf")
        print(
            f"\nRelative performance: {'PyTorch' if speedup < 1 else 'JAX'} is {max(speedup, 1/speedup):.2f}x faster"
        )
    else:
        print("PyTorch times reported in logs above (using abstract_training_pipeline)")

    print("=" * 60)


def quick_test_compare():
    """Run a quick comparison test with fewer iterations."""
    config = BenchmarkConfig(
        num_iterations=10, val_loss_every=5, warmup_steps=3, num_runs=1
    )
    compare_training(config)


def medium_test_compare():
    """Run a medium-length comparison test."""
    config = BenchmarkConfig(
        num_iterations=50, val_loss_every=10, warmup_steps=5, num_runs=1
    )
    compare_training(config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "compare":
            compare_training()
        elif command == "jax":
            train_jax_model()
        elif command == "torch" or command == "pytorch":
            train_pytorch_model()
        elif command == "quick":
            quick_test_compare()
        elif command == "medium":
            medium_test_compare()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: compare, jax, torch, quick, medium")
    else:
        # Default to PyTorch
        train_pytorch_model()
