import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from aim import Run
from data_utils import data_generator
from nanodo_model import DoConfig, TransformerDo
from typing import Dict

def get_optimizer(config: Dict):
    """Create optimizer from configuration dictionary.
    
    Args:
        config: Dictionary containing optimizer configuration with:
            - name: Optimizer name ('adamw', 'adam', 'sgd', 'rmsprop')
            - learning_rate: Base learning rate
            - Other optimizer-specific parameters
    
    Returns:
        Tuple of (optax optimizer instance, cleaned config dict with only used params)
    """
    name = config['name'].lower()
    lr = config['learning_rate']
    
    if name == 'adamw':
        cleaned_config = {
            'name': name,
            'learning_rate': lr,
            'b1': config.get('b1', 0.9),
            'b2': config.get('b2', 0.999),
            'eps': config.get('eps', 1e-8),
            'weight_decay': config.get('weight_decay', 0.01)
        }
        return optax.adamw(
            learning_rate=lr,
            b1=config.get('b1', 0.9),
            b2=config.get('b2', 0.999),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0.01)
        )
    elif name == 'adam':
        cleaned_config = {
            'name': name,
            'learning_rate': lr,
            'b1': config.get('b1', 0.9),
            'b2': config.get('b2', 0.999),
            'eps': config.get('eps', 1e-8)
        }
        return optax.adam(
            learning_rate=lr,
            b1=config.get('b1', 0.9),
            b2=config.get('b2', 0.999),
            eps=config.get('eps', 1e-8)
        )
    elif name == 'sgd':
        cleaned_config = {
            'name': name,
            'learning_rate': lr,
            'momentum': config.get('momentum', 0.0),
            'nesterov': config.get('nesterov', False)
        }
        return optax.sgd(
            learning_rate=lr,
            momentum=config.get('momentum', 0.0),
            nesterov=config.get('nesterov', False)
        )
    elif name == 'rmsprop':
        cleaned_config = {
            'name': name,
            'learning_rate': lr,
            'decay': config.get('decay', 0.9),
            'eps': config.get('eps', 1e-8),
            'momentum': config.get('momentum', 0.0)
        }
        return optax.rmsprop(
            learning_rate=lr,
            decay=config.get('decay', 0.9),
            eps=config.get('eps', 1e-8),
            momentum=config.get('momentum', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def main():
    # Initialize Aim run
    run = Run()
    
    # Configuration
    batch_size = 8
    seq_len = 64
    vocab_size = 16  # Increased to 16 for more interesting patterns
    learning_rate = 1e-3
    num_steps = 1200
    eval_every = num_steps / 24
    seed = 230

    # Optimizer configuration
    optimizer_config = {
        'name': 'sgd',
        'learning_rate': learning_rate,
        'momentum': 0.0,
        'nesterov': False
    }

    # Initialize optimizer and get cleaned config
    optimizer, cleaned_config = get_optimizer(optimizer_config)

    # Track hyperparameters
    run['hparams'] = {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'vocab_size': vocab_size,
        'learning_rate': learning_rate,
        'num_steps': num_steps,
        'eval_every': eval_every,
        'seed': seed,
        'model_dim': 32,
        'n_heads': 4,
        'n_layers': 4,
        'ff_dim': 32,
        'optimizer': cleaned_config
    }

    # Initialize model
    config = DoConfig(
        D=run['hparams']['model_dim'],          # Model dimension
        H=run['hparams']['n_heads'],           # Number of attention heads
        L=seq_len,     # Max sequence length
        N=run['hparams']['n_layers'],           # Number of layers
        V=vocab_size,  # Vocabulary size
        F=run['hparams']['ff_dim']          # Feed-forward dimension
    )
    model = TransformerDo(config)

    # Initialize optimizer and get cleaned config
    optimizer, cleaned_config = get_optimizer(optimizer_config)
    
    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    init_rng, data_rng = jax.random.split(rng)

    # Initialize model parameters
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input)
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Loss and accuracy function
    def compute_metrics(params, batch):
        inputs = batch[:, :-1]  # All tokens except last
        targets = batch[:, 1:]   # All tokens except first
        
        # Get model predictions
        logits = model.apply(params, inputs)
        
        # Calculate cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=targets
        )
        loss = jnp.mean(loss)
        
        # Calculate accuracy
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == targets)
        
        return loss, accuracy

    # Training step
    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, accuracy), grads = jax.value_and_grad(compute_metrics, has_aux=True)(params, batch)
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
            # Generate evaluation batch with new RNG
            _, eval_rng = jax.random.split(data_rng)
            eval_batch = data_generator(eval_rng, batch_size, seq_len, vocab_size)
            eval_loss, eval_acc = evaluate(params, eval_batch)
            
            tqdm.write(
                f"Step {step:4d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | "
                f"Eval Acc: {eval_acc:.4f}"
            )
            
            # Track metrics with Aim
            run.track(train_loss, name='train_loss', step=step, context={'subset': 'train'})
            run.track(train_acc, name='accuracy', step=step, context={'subset': 'train'})
            run.track(eval_loss, name='val_loss', step=step, context={'subset': 'val'})
            run.track(eval_acc, name='accuracy', step=step, context={'subset': 'val'})

    # Final test
    test_batch = data_generator(jax.random.PRNGKey(0), 5, seq_len, vocab_size)
    inputs = test_batch[:, :-1]
    targets = test_batch[:, 1:]
    predictions = jnp.argmax(model.apply(params, inputs), axis=-1)
    
    print("\nSample predictions:")
    for i in range(3):
        print(f"Input:  {inputs[i, :10].tolist()}...")
        print(f"Target: {targets[i, :10].tolist()}...")
        print(f"Pred:   {predictions[i, :10].tolist()}...")
        print()

if __name__ == "__main__":
    main()
