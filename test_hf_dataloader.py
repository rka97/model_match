from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Tuple


def get_hf_dataloader(
    batch_size: int = 8, seq_len: int = 32, framework: str = "torch", split="train"
):
    """
    Create a data loader from HuggingFace's FineWeb dataset.

    Args:
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence
        framework: Either "torch" or "jax" to specify output tensor type
    """
    # Initialize tokenizer and get vocab size
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    vocab_size = tokenizer.vocab_size
    # Load the FineWeb dataset in streaming mode
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split=split,
        streaming=False,
        cache_dir="/home/ak4605/data/",
    )
    fw = fw.batch(batch_size=batch_size, drop_last_batch=True)

    def batch_iterator():
        for doc in fw:
            if framework == "torch":
                # Tokenize document text
                _tokenize = lambda x: tokenizer(x, return_tensors="pt")[
                    "input_ids"
                ].squeeze()
                token_ids = torch.stack(
                    [_tokenize(x)[: seq_len + 1] for x in doc["text"]]
                )
                # Take first seq_len+1 tokens and convert to one-hot
                tokens = F.one_hot(token_ids, num_classes=vocab_size).float()
                # Split into input/target
                inputs, targets = tokens[:, :-1, :], tokens[:, 1:, :]
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
            elif framework == "jax":
                _tokenize = lambda x: tokenizer(x, return_tensors="jax")[
                    "input_ids"
                ].squeeze()
                token_ids = jnp.stack(
                    [_tokenize(x)[: seq_len + 1] for x in doc["text"]]
                )
                tokens = jax.nn.one_hot(token_ids, num_classes=vocab_size)
                inputs, targets = tokens[:, :-1], tokens[:, 1:]
                devices = jax.devices("gpu")
                inputs, targets = jax.device_put(inputs), jax.device_put(targets)
            yield inputs, targets

    return batch_iterator()


def test_hf_dataloader(framework: str = "torch"):
    """
    Test the HuggingFace data loader for a given framework.
    """
    batch_size = 8
    seq_len = 32
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    vocab_size = tokenizer.vocab_size

    print(f"\nTesting HuggingFace {framework} data loader...")

    # Get the data loader
    data_loader = get_hf_dataloader(batch_size, seq_len, framework)

    # Get first batch
    inputs, targets = next(data_loader)

    # Print info
    print(f"Inputs shape: {inputs.shape}")
    print(f"Inputs sample: {inputs[0] if framework == 'torch' else inputs[0][0]}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets sample: {targets[0] if framework == 'torch' else targets[0][0]}")

    # Verify shapes and content
    if framework == "torch":
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert inputs.shape == (batch_size, seq_len, vocab_size)
        assert targets.shape == (batch_size, seq_len, vocab_size)
        assert torch.all(
            inputs[:, 1:] == targets[:, :-1]
        ), "Targets should be shifted inputs"
    elif framework == "jax":
        assert isinstance(inputs, jax.Array)
        assert isinstance(targets, jax.Array)
        assert inputs.shape == (batch_size, seq_len, vocab_size)
        assert targets.shape == (batch_size, seq_len, vocab_size)
        assert jnp.all(
            inputs[:, 1:] == targets[:, :-1]
        ), "Targets should be shifted inputs"
    print(f"HuggingFace {framework} data loader test passed!")


if __name__ == "__main__":
    # Test both torch and jax versions
    test_hf_dataloader(framework="torch")
    test_hf_dataloader(framework="jax")
