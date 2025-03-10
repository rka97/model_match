import os
from pathlib import Path
import torch
import jax
import jax.numpy as jnp
from abstract_training_pipeline import distributed_data_generator

def test_dataloader(framework="torch"):
    # Setup mock environment variables
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    
    # Use finewebedu data files
    data_files = "data/finewebedu_train_*.bin"
    batch_size = 8
    
    # Test the data loader
    data_gen = distributed_data_generator(data_files, batch_size, rank=0, world_size=1, framework=framework)
    
    # Get first batch
    inputs, targets = next(data_gen)
    
    print(f"\nTesting {framework} data loader...")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Inputs sample: {inputs[:8]}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets sample: {targets[:8]}")
    
    # Verify shapes and content
    if framework == "torch":
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert inputs.shape == (batch_size,)
        assert targets.shape == (batch_size,)
        assert torch.all(inputs[1:] == targets[:-1]), "Targets should be shifted inputs"
    elif framework == "jax":
        # Verify we got sharded arrays
        assert isinstance(inputs, jax.Array)  # List of DeviceArray
        assert isinstance(targets, jax.Array)
        
        # Check shapes of individual shards
        shard_size = batch_size // len(jax.devices("gpu"))
        assert inputs.shape == (len(jax.devices("gpu")), shard_size)
        assert targets.shape == (len(jax.devices("gpu")), shard_size)

        # Verify shifted relationship by concatenating shards
        all_inputs = jnp.concatenate([jax.device_get(inp) for inp in inputs])
        all_targets = jnp.concatenate([jax.device_get(tgt) for tgt in targets])
        assert jnp.all(all_inputs[1:] == all_targets[:-1]), "Targets should be shifted inputs"
        
        # Verify devices are correct
        devices = jax.devices("gpu")

    print(f"{framework.capitalize()} data loader test passed!")

if __name__ == "__main__":
    # Test both torch and jax versions
    test_dataloader(framework="torch")
    test_dataloader(framework="jax")
