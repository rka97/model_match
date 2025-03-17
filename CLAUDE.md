# Model Match Project Commands & Style Guide

## Build/Test Commands
- Run Python in JAX environment: `mamba activate jax-env && python [script].py`
- Run single test: `mamba activate jax-env && python test_dataloader.py`

## Training Commands
- Train PyTorch model: `mamba activate jax-env && python trainer.py` or `python trainer.py torch`
- Train JAX model: `mamba activate jax-env && python trainer.py jax`
- Train on sequential data (PyTorch): `mamba activate jax-env && python trainer.py sequential pytorch`
- Train on sequential data (JAX): `mamba activate jax-env && python trainer.py sequential jax`

## Benchmarking Commands
- Full comparison: `mamba activate jax-env && python trainer.py compare`
  - Runs 100 iterations per framework with 2 runs for averaging
- Medium test: `mamba activate jax-env && python trainer.py medium`
  - Medium benchmark with 50 iterations and 1 run
- Quick test: `mamba activate jax-env && python trainer.py quick`
  - Faster, shorter benchmark with 10 iterations and 1 run
  
Both frameworks use:
- Identical model architecture (simple 2-layer MLP)
- Same dataloader and AdamW optimizer
- Warmup steps to exclude compilation overhead
- PyTorch with torch.compile() (reduce-overhead mode)
- JAX with JIT compilation and optimized functions

## Code Style Guidelines
- **Imports**: Standard order - built-in, third-party, local modules
- **Types**: Use type hints for function parameters and returns
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Docstrings**: Required for classes and functions, include param descriptions
- **Exception Handling**: Specific exceptions with context messages
- **PyTorch/JAX**: Support both frameworks where possible, use framework param
- **Framework compat**: Check for tensor types with framework-specific asserts

## Project Structure
- Model components in individual files (match_*.py)
- Training logic in abstract_training_pipeline.py
- Data in data/ directory (binary token format)
- Logs saved to logs/ directory