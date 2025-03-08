#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the training framework
from abstract_training_pipeline import AbstractModel, train, Hyperparameters

class SimpleLanguageModel(AbstractModel):
    """
    A basic language model with just two linear layers.
    """
    def __init__(self, vocab_size=50257, hidden_dim=256, embedding_dim=384):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Two linear layers
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

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

def train_simple_model():
    """
    Training function that uses PyTorch's AdamW optimizer.
    """
    # Set up hyperparameters
    args = Hyperparameters()
    args.train_files = "data/finewebedu_train_*.bin"  # Update to your actual data path
    args.val_files = "data/finewebedu_val_*.bin"      # Update to your actual data path
    args.train_seq_len = 32      # Smaller sequence length for this simple model
    args.val_seq_len = 32        # Smaller sequence length for validation
    args.val_tokens = 32 * 32       # Number of tokens to validate on
    args.num_iterations = 500     # Fewer iterations for testing
    args.val_loss_every = 50      # Validate every 50 steps

    # Create a custom AdamW optimizer wrapper that follows the AbstractOptimizer interface
    class AdamWWrapper:
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
            # Group parameters by size for efficiency
            param_groups = []
            for p in params:
                # Only include parameters that require gradients
                if p.requires_grad:
                    param_groups.append(p)

            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
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

    # Train the model
    train(SimpleLanguageModel, [AdamWWrapper], args)

if __name__ == "__main__":
    train_simple_model()
