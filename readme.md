# README


The goal of this project is to have a new model, implemented in both JAX and PyTorch, that has (a) a matching architecture in both frameworks and (b) roughly similar speeds.

# Comparison Table

| Layer/Component | PyTorch (Niccolo’s) | Flax (NanoDo) | Notes | Decision |
| :---- | :---- | :---- | :---- | :---- |
| Normalization | RMSNorm  | LayerNorm | Both pre-norm | RMSNorm |
| Positional Encoding | Rotary (RoPE)  | Additive |  | Rotary |
| Attention QKV | Single projection | Separate projections | Equiv function, might be different in speed (matmul) | Single fused projection, also try QK norm |
| Attention Calculation | Pytorch’s functional api  \`scaled\_dot\_product\_attention\`  | Manual einsum | PyTorch uses optimized kernel when available | Try flax attention, if it doesn’t match use einsum for both |
| Feed-Forward | MLP/GLU/ReLU² variants | MLP with GELU | GLU | GLU |
| Activation | SiLU (Swish) | GELU | Also try SwiGLU | Swish (aka SwiGLU) |
| Weight Tying | Direct sharding | Embed.attend |  | Try untied |
| Initialization | Normal \+ bias at 0s \+ residual scaling \+ all linear to std 0.02 | Xavier uniform | Different weight initialization strategies | Xavier Uniform \+ residual scaling (worth trying) |
|  |  |  |  |  |
|  |  |  |  |  |
| Embedding Initialization \- input | std 0.02 | Jax code :  nn.initializers.variance\_scaling(       1.0, 'fan\_in', 'normal', out\_axis=0) |  |  |
|  |  |  |  | std 0.02 |
|  |  |  |  |  |
| Embedding Initialization \- output  | std 0.02 | Jax code :  nn.initializers.variance\_scaling(       1.0, 'fan\_in', 'normal', out\_axis=0) |  | Add to configuration: which initializer to use, the std deviation, and whether to use tying |
|  |  |  |  |  |
|  |  |  |  |  |
| Residual Scaling | Yes (depth-based) | No | PyTorch scales by 1/sqrt(2\*n\_layers) for stability at init |  |
|  |  |  |  | Yes (maybe) |
|  |  |  |  |  |
| ~~Memory Optimization~~ | ~~PyTorch FSDP~~ | ~~Rematerialization \+ FSDP~~ | ~~DeepMind includes explicit memory optimizations, also opt sharding~~ |  |
|  |  |  |  |  |

References for design choices:

* SwiGLU: Shazeer, GLU Variants Improve Transformer [https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)  
* RoPE:   
* RMSNorm:   
* Residual Branch scaling: GPT-2 paper  
* Fused QKV projection: faster matmul

Some more design choices that we’ll eventually have to make:

* Model size  
* Token budget: follow Chinchilla  
* Tokenizer  
* Context length
