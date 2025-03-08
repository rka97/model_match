# Based on the modded-nano-gpt repo from KellerJ
import os
import sys
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn

# -----------------------------------------------------------------------------
# Basic Data Loader for distributed training

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int):
    """
    Generator for distributed training data.
    Yields (inputs, targets) pairs for training a language model.
    """
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)  # use itertools.cycle(files) instead for multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0

    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# Abstract Model class

class AbstractModel(nn.Module):
    """
    Abstract base class for models.
    Implement this for your specific model architecture.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_seq, target_seq):
        """
        Override this method in your model implementation.
        Should return the loss for the current batch.
        """
        raise NotImplementedError("Subclasses must implement forward")

# -----------------------------------------------------------------------------
# Abstract Optimizer class

class AbstractOptimizer:
    """
    Abstract base class for optimizers.
    Implement this for your specific optimization algorithm.
    """
    def __init__(self, params):
        self.params = list(params)
        self.param_groups = [{'params': self.params, 'lr': 0.01}]

    def zero_grad(self, set_to_none=False):
        for param in self.params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def step(self):
        """
        Override this method in your optimizer implementation.
        Should perform one optimization step based on gradients.
        """
        raise NotImplementedError("Subclasses must implement step")

    def state_dict(self):
        """
        Return a state dict for checkpointing.
        """
        return {'param_groups': self.param_groups}

    def load_state_dict(self, state_dict):
        """
        Load a state dict from a checkpoint.
        """
        self.param_groups = state_dict['param_groups']

# -----------------------------------------------------------------------------
# Training framework

@dataclass
class Hyperparameters:
    # data
    train_files: str = "data/finewebedu_train_*.bin"  # input .bin to train on
    val_files: str = "data/finewebedu_val_*.bin"  # input .bin to eval validation loss on
    val_tokens: int = 32 * 64  # how many tokens of validation data to use
    train_seq_len: int = 32  # sequence length for training
    val_seq_len: int = 32  # sequence length for validation
    # optimization
    num_iterations: int = 1000  # number of iterations to run
    cooldown_frac: float = 0.4  # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every: int = 100  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = False

def train(model_class, optimizer_classes, args=None):
    """
    Main training function.

    Args:
        model_class: Class for the model to train
        optimizer_classes: List of optimizer classes to use
        args: Hyperparameters for training
    """
    if args is None:
        args = Hyperparameters()

    # torchrun sets these env variables
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device_id = int(os.environ.get("LOCAL_RANK", "0"))

    assert torch.cuda.is_available()
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)

    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = (rank == 0)  # this process will do logging, checkpointing etc.

    # Set up logging
    logfile = None
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)

    def print0(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)

    # Log information about the environment
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

    # Construct model and optimizer
    model = model_class().to(device)
    if world_size > 1:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)

    # Init optimizers
    optimizers = [opt_cls(model.parameters()) for opt_cls in optimizer_classes]

    # Learning rate schedule
    def get_lr(step):
        x = step / args.num_iterations  # progress in training
        assert 0 <= x < 1
        if x < 1 - args.cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / args.cooldown_frac
            return w * 1.0 + (1 - w) * 0.1

    # Optional: Compile the model
    if hasattr(torch, 'compile'):
        model = torch.compile(model, dynamic=False)

    # Warmup
    warmup_steps = 10
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]
    )

    vocab_size = getattr(model, 'vocab_size', 50000)  # Default if not specified

    for _ in range(warmup_steps):
        inputs = targets = torch.randint(0, vocab_size, size=(args.train_seq_len,), device=device)
        model(inputs, targets).backward()

        if world_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        for opt in optimizers:
            opt.step()

        model.zero_grad(set_to_none=True)

    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state

    # Training loop
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
    training_time_ms = 0

    # Start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_steps = args.num_iterations
    for step in range(train_steps + 1):
        last_step = (step == train_steps)

        # Validation
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # Stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()

            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
            val_loss = 0

            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets)

            val_loss /= val_steps
            del val_loader

            if world_size > 1:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} "
                  f"train_time:{training_time_ms:.0f}ms "
                  f"step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

            model.train()

            # Start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(
                    step=step,
                    model=model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers]
                )
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            break

        # Training step
        inputs, targets = next(train_loader)
        model(inputs, targets).backward()

        if world_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        # Set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                if hasattr(group, "initial_lr"):
                    group["lr"] = group["initial_lr"] * get_lr(step)

        # Step the optimizers
        for opt in optimizers:
            opt.step()

        # Zero the gradients
        model.zero_grad(set_to_none=True)

        # Logging
        if step % 10 == 0:  # Log less frequently
            approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f"step:{step+1}/{train_steps} "
                  f"train_time:{approx_training_time_ms:.0f}ms "
                  f"step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

    if world_size > 1:
        dist.destroy_process_group()

# Example usage commented out to avoid running when imported
# if __name__ == "__main__":
#     # Define your model that inherits from AbstractModel
#     class MyModel(AbstractModel):
#         def __init__(self):
#             super().__init__()
#             # Your model initialization here
#
#         def forward(self, input_seq, target_seq):
#             # Your model forward pass here
#             return torch.tensor(0.0, device='cuda')  # Placeholder
#
#     # Define your optimizer class that inherits from AbstractOptimizer
#     class MyOptimizer(AbstractOptimizer):
#         def __init__(self, params):
#             super().__init__(params)
#             # Your optimizer initialization here
#
#         def step(self):
#             # Your optimization step here
#             pass
#
#     # Example of how to use the training framework
#     args = Hyperparameters()
#     args.train_files = "data/fineweb10B/fineweb_train_*.bin"
#     args.val_files = "data/fineweb10B/fineweb_val_*.bin"
#     args.num_iterations = 1000
#
#     # Train with your model and optimizer
#     # train(MyModel, [MyOptimizer], args)
