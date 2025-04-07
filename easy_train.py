import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from copy import deepcopy
from typing import List, Any
from flax.traverse_util import flatten_dict, unflatten_dict
from one_sided_shampoo import one_sided_shampoo
from pathlib import Path
from tqdm import tqdm
from aim import Run
from data_utils import data_generator
from nanodo_model import DoConfig, TransformerDo
from typing import Dict, Tuple, Any


def get_optimizer(config: Dict) -> Tuple[optax.GradientTransformation, Dict]:
    """Create optimizer from configuration dictionary.

    Args:
        config: Dictionary containing optimizer configuration with:
            - name: Optimizer name ('adamw', 'adam', 'sgd', 'rmsprop')
            - learning_rate: Base learning rate
            - Other optimizer-specific parameters

    Returns:
        Tuple of (optax optimizer instance, cleaned config dict with only used params)
    """
    name = config["name"].lower()
    lr = config["learning_rate"]

    if name == "adamw":
        cleaned_config = {
            "name": name,
            "learning_rate": lr,
            "b1": config.get("b1", 0.9),
            "b2": config.get("b2", 0.999),
            "eps": config.get("eps", 1e-8),
            "weight_decay": config.get("weight_decay", 0.01),
        }
        return (
            optax.adamw(
                learning_rate=cleaned_config["learning_rate"],
                b1=cleaned_config["b1"],
                b2=cleaned_config["b2"],
                eps=cleaned_config["eps"],
                weight_decay=cleaned_config["weight_decay"],
            ),
            cleaned_config,
        )
    elif name == "adam":
        cleaned_config = {
            "name": name,
            "learning_rate": lr,
            "b1": config.get("b1", 0.9),
            "b2": config.get("b2", 0.999),
            "eps": config.get("eps", 1e-8),
        }
        return (
            optax.adam(
                learning_rate=cleaned_config["learning_rate"],
                b1=cleaned_config["b1"],
                b2=cleaned_config["b2"],
                eps=cleaned_config["eps"],
            ),
            cleaned_config,
        )
    elif name == "sgd":
        cleaned_config = {
            "name": name,
            "learning_rate": lr,
            "momentum": config.get("momentum", 0.0),
            "nesterov": config.get("nesterov", False),
        }
        return (
            optax.sgd(
                learning_rate=cleaned_config["learning_rate"],
                momentum=cleaned_config["momentum"],
                nesterov=cleaned_config["nesterov"],
            ),
            cleaned_config,
        )
    elif name == "rmsprop":
        cleaned_config = {
            "name": name,
            "learning_rate": lr,
            "decay": config.get("decay", 0.9),
            "eps": config.get("eps", 1e-8),
            "momentum": config.get("momentum", 0.0),
        }
        return (
            optax.rmsprop(
                learning_rate=cleaned_config["learning_rate"],
                decay=cleaned_config["decay"],
                eps=cleaned_config["eps"],
                momentum=cleaned_config["momentum"],
            ),
            cleaned_config,
        )
    elif name == "muon":
        cleaned_config = {
            "name": name,
            "learning_rate": lr,
            "ns_coeffs": config.get("ns_coeffs", (3.4445, -4.775, 2.0315)),
            "ns_steps": config.get("ns_steps", 5),
            "beta": config.get("beta", 0.95),
            "eps": config.get("eps", 1e-8),
            "nesterov": config.get("nesterov", True),
            "adaptive": config.get("adaptive", False),
            "adam_b1": config.get("adam_b1", 0.9),
            "adam_b2": config.get("adam_b2", 0.999),
            "adam_eps_root": config.get("adam_eps_root", 0.0),
            "adam_weight_decay": config.get("adam_weight_decay", 0.0),
        }
        return (
            optax.contrib.muon(
                learning_rate=cleaned_config["learning_rate"],
                ns_coeffs=cleaned_config["ns_coeffs"],
                ns_steps=cleaned_config["ns_steps"],
                beta=cleaned_config["beta"],
                eps=cleaned_config["eps"],
                nesterov=cleaned_config["nesterov"],
                adaptive=cleaned_config["adaptive"],
                adam_b1=cleaned_config["adam_b1"],
                adam_b2=cleaned_config["adam_b2"],
                adam_eps_root=cleaned_config["adam_eps_root"],
                adam_weight_decay=cleaned_config["adam_weight_decay"],
            ),
            cleaned_config,
        )
    elif name == "lion":
        cleaned_config = {
            "name": name,
            "learning_rate": lr,
            "b1": config.get("b1", 0.9),
            "b2": config.get("b2", 0.99),
            "mu_dtype": config.get("mu_dtype", None),
            "weight_decay": config.get("weight_decay", 0.001),
        }
        return (
            optax.lion(
                learning_rate=cleaned_config["learning_rate"],
                b1=cleaned_config["b1"],
                b2=cleaned_config["b2"],
                mu_dtype=cleaned_config["mu_dtype"],
                weight_decay=cleaned_config["weight_decay"],
            ),
            cleaned_config,
        )
    elif name == "one_sided_shampoo":
        cleaned_config = {
            "name": name,
            "learning_rate": lr,
            "beta": config.get("beta", 0.9),
            "epsilon": config.get("epsilon", 1e-8),
            "mu_dtype": config.get("mu_dtype", None),
            "adam_b1": config.get("adam_b1", 0.9),
            "adam_b2": config.get("adam_b2", 0.999),
            "adam_eps_root": config.get("adam_eps_root", 0.0),
            "adam_weight_decay": config.get("adam_weight_decay", 0.0),
        }
        return (
            one_sided_shampoo(
                learning_rate=cleaned_config["learning_rate"],
                beta=cleaned_config["beta"],
                epsilon=cleaned_config["epsilon"],
                mu_dtype=cleaned_config["mu_dtype"],
                adam_b1=cleaned_config["adam_b1"],
                adam_b2=cleaned_config["adam_b2"],
                adam_eps_root=cleaned_config["adam_eps_root"],
                adam_weight_decay=cleaned_config["adam_weight_decay"],
            ),
            cleaned_config,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def parse_grid_value(value: Any) -> List[Any]:
    """Parse grid search notation like (start, end, num, spacing)."""
    if not isinstance(value, str) or not (
        value.startswith("(") and value.endswith(")")
    ):
        return [value]

    # Parse grid spec (start, end, num, spacing)
    parts = value[1:-1].split(",")
    if len(parts) != 4:
        return [value]

    try:
        start = float(parts[0].strip())
        end = float(parts[1].strip())
        num = int(parts[2].strip())
        spacing = parts[3].strip().lower()
    except (ValueError, IndexError):
        return [value]

    if spacing == "log":
        return list(np.logspace(np.log10(start), np.log10(end), num))
    elif spacing == "lin":
        return list(np.linspace(start, end, num))
    else:
        return [value]


def expand_grid_config(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a config with grid values into multiple configs."""
    from itertools import product

    # Find all grid parameters
    grid_params = {}
    flat_config = flatten_dict(base_config)
    for key, value in flat_config.items():
        grid_values = parse_grid_value(value)
        if len(grid_values) > 1:
            grid_params[key] = grid_values

    if not grid_params:
        return [base_config]

    # Generate all combinations
    grid_combinations = product(*grid_params.values())

    # Build all config variants
    configs = []
    for combo in grid_combinations:
        new_config = deepcopy(base_config)
        flat_new = flatten_dict(new_config)
        for (key, _), value in zip(grid_params.items(), combo):
            flat_new[key] = value
        configs.append(unflatten_dict(flat_new))

    return configs


def load_config(config_path: str = "config.yml") -> List[Dict[str, Any]]:
    """Load configuration from YAML file, expanding any grid search parameters."""
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    return expand_grid_config(base_config)


def train_model(
    model: TransformerDo,
    params: Dict,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    train_config: Dict,
    eval_dataset: jnp.ndarray,
    run: Run,
) -> Tuple[Dict, optax.OptState]:
    """Train the model with given configuration.

    Args:
        model: Initialized model
        params: Model parameters
        optimizer: Optax optimizer
        opt_state: Optimizer state
        train_config: Training configuration dictionary
        eval_dataset: Fixed evaluation dataset
        run: Aim run for tracking

    Returns:
        Tuple of (trained parameters, final optimizer state)
    """
    batch_size = train_config["batch_size"]
    seq_len = train_config["seq_len"]
    num_steps = train_config["num_steps"]
    eval_every = train_config["eval_every"]
    seed = train_config["seed"]

    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    _, data_rng = jax.random.split(rng)

    # Loss and accuracy function
    def compute_metrics(params, batch):
        inputs = batch[:, :-1]  # All tokens except last
        targets = batch[:, 1:]  # All tokens except first

        # Get model predictions
        logits = model.apply(params, inputs)

        # Calculate cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=targets
        )
        loss = jnp.mean(loss)

        # Calculate accuracy
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == targets)

        return loss, accuracy

    # Training step
    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, accuracy), grads = jax.value_and_grad(compute_metrics, has_aux=True)(
            params, batch
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, accuracy

    # Evaluation function
    @jax.jit
    def evaluate(params, batch):
        return compute_metrics(params, batch)

    # Training loop
    for step in tqdm(range(num_steps), desc="Training"):
        # Generate training batch
        data_rng, batch_rng = jax.random.split(data_rng)
        batch = data_generator(batch_rng, batch_size, seq_len)

        # Training step
        params, opt_state, train_loss, train_acc = train_step(params, opt_state, batch)

        # Periodic evaluation
        if step % eval_every == 0 or step == num_steps - 1:
            eval_loss, eval_acc = evaluate(params, eval_dataset)

            tqdm.write(
                f"Step {step:4d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | "
                f"Eval Acc: {eval_acc:.4f}"
            )

            # Track metrics with Aim
            run.track(
                train_loss, name="train_loss", step=step, context={"subset": "train"}
            )
            run.track(
                train_acc, name="accuracy", step=step, context={"subset": "train"}
            )
            run.track(eval_loss, name="val_loss", step=step, context={"subset": "val"})
            run.track(eval_acc, name="accuracy", step=step, context={"subset": "val"})

    return params, opt_state


def run_experiment(config: Dict[str, Any]):
    # Training config
    training_cfg = config["training"]
    batch_size = training_cfg["batch_size"]
    eval_batch_size = training_cfg.get(
        "eval_batch_size", batch_size * 4
    )  # Larger eval batches
    seq_len = training_cfg["seq_len"]
    vocab_size = training_cfg["vocab_size"]
    num_steps = training_cfg["num_steps"]
    eval_every = training_cfg["eval_every"]
    seed = training_cfg["seed"]

    # Generate fixed evaluation dataset
    _, eval_rng = jax.random.split(
        jax.random.PRNGKey(seed + 1)
    )  # Different seed than training
    eval_dataset = data_generator(eval_rng, eval_batch_size, seq_len, vocab_size)

    # Model config
    model_cfg = config["model"]

    # Optimizer configuration
    optimizer_config = config["optimizer"]
    optimizer, cleaned_config = get_optimizer(optimizer_config)

    # Initialize Aim run
    import time

    experiment_name = f"{optimizer_config['name']}_{int(time.time())}"
    run = Run(experiment=experiment_name)

    # Track hyperparameters
    run["hparams"] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "num_steps": num_steps,
        "eval_every": eval_every,
        "seed": seed,
        "model_dim": model_cfg["dim"],
        "n_heads": model_cfg["n_heads"],
        "n_layers": model_cfg["n_layers"],
        "ff_dim": model_cfg["ff_dim"],
        "optimizer": cleaned_config,
    }

    # Initialize model
    config = DoConfig(
        D=run["hparams"]["model_dim"],  # Model dimension
        H=run["hparams"]["n_heads"],  # Number of attention heads
        L=seq_len,  # Max sequence length
        N=run["hparams"]["n_layers"],  # Number of layers
        V=vocab_size,  # Vocabulary size
        F=run["hparams"]["ff_dim"],  # Feed-forward dimension
    )
    model = TransformerDo(config)

    # Initialize optimizer and get cleaned config
    optimizer, cleaned_config = get_optimizer(optimizer_config)

    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    init_rng, _ = jax.random.split(rng)

    # Initialize model parameters
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input)

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Train the model
    params, opt_state = train_model(
        model=model,
        params=params,
        optimizer=optimizer,
        opt_state=opt_state,
        train_config=training_cfg,
        eval_dataset=eval_dataset,
        run=run,
    )

    # Final test on our fixed eval dataset
    test_batch = eval_dataset[:5]  # Just show first 5 examples
    inputs = test_batch[:, :-1]
    targets = test_batch[:, 1:]
    predictions = jnp.argmax(model.apply(params, inputs), axis=-1)

    print("\nSample predictions:")
    for i in range(3):
        print(f"Input:  {inputs[i, :10].tolist()}...")
        print(f"Target: {targets[i, :10].tolist()}...")
        print(f"Pred:   {predictions[i, :10].tolist()}...")
        print()


def main():
    """Run all experiments from config, handling grid search if specified."""
    configs = load_config()

    for i, config in enumerate(configs):
        print(f"\n=== Running experiment {i+1}/{len(configs)} ===")
        if len(configs) > 1:
            print("Config:", config)
        run_experiment(config)


if __name__ == "__main__":
    main()
