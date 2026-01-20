---
description: Deep ML engineering knowledge including debugging workflows (NaN, OOM, slow training), performance profiling (PyTorch profiler, Nsight), common pitfalls, architecture decisions, and memory estimation. Reference for advanced troubleshooting.
globs: ["*.py", "**/ml/**", "**/models/**", "**/training/**"]
alwaysApply: false
---

# Expert ML Engineering Reference

Deep knowledge for advanced practitioners: debugging, profiling, pitfalls, and architecture decisions.

## Performance Profiling Workflow

### 1. Identify the Bottleneck Type
```
Compute-bound: GPU utilization high, memory bandwidth low
Memory-bound: Memory bandwidth saturated, compute underutilized
Latency-bound: Neither saturated, kernels too small or too many launches
```

### 2. PyTorch Profiler
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("model_inference"):
        output = model(input)

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
```

### 3. NVIDIA Nsight Systems
```bash
# Profile entire application
nsys profile -o report python train.py

# Key metrics to check:
# - Kernel execution time
# - Memory transfer time
# - CPU/GPU overlap
# - Kernel launch overhead
```

### 4. Roofline Analysis
```
Arithmetic Intensity = FLOPs / Bytes Transferred

If AI < ridge point → Memory-bound (optimize memory access)
If AI > ridge point → Compute-bound (optimize compute)

# Example: Matrix multiply C = A @ B
# FLOPs = 2 * M * N * K
# Bytes = (M*K + K*N + M*N) * bytes_per_element
```

## Common Pitfalls & Anti-Patterns

### PyTorch Pitfalls

**1. Forgetting torch.no_grad() for inference**
```python
# Bad - wastes memory tracking gradients
output = model(input)

# Good
with torch.no_grad():
    output = model(input)

# Best for inference
with torch.inference_mode():
    output = model(input)
```

**2. Data transfer bottleneck**
```python
# Bad - transfers every batch
for batch in loader:
    batch = batch.to(device)  # Slow!
    output = model(batch)

# Good - use DataLoader with pin_memory
loader = DataLoader(dataset, pin_memory=True, num_workers=4)
for batch in loader:
    batch = batch.to(device, non_blocking=True)
```

**3. Python loop over tensor elements**
```python
# Bad - 1000x slower
result = torch.zeros(n)
for i in range(n):
    result[i] = tensor[i] * 2

# Good - vectorized
result = tensor * 2
```

**4. Creating tensors in training loop**
```python
# Bad - allocates memory every iteration
for batch in loader:
    mask = torch.ones(batch.size(0), device=device)
    
# Good - pre-allocate and reuse
mask = torch.ones(max_batch_size, device=device)
for batch in loader:
    current_mask = mask[:batch.size(0)]
```

**5. Not using channels_last for CNNs**
```python
# Good - 20-30% speedup on modern GPUs
model = model.to(memory_format=torch.channels_last)
input = input.to(memory_format=torch.channels_last)
```

### CUDA Pitfalls

**1. Uncoalesced memory access**
```cpp
// Bad - strided access, each thread hits different cache line
output[threadIdx.x * stride] = input[threadIdx.x * stride];

// Good - coalesced, consecutive threads access consecutive memory
output[threadIdx.x] = input[threadIdx.x];
```

**2. Shared memory bank conflicts**
```cpp
// Bad - all threads access same bank
__shared__ float smem[32][32];
float val = smem[threadIdx.x][0];  // Column access = conflicts

// Good - add padding to avoid conflicts
__shared__ float smem[32][33];  // Extra column
float val = smem[threadIdx.x][0];
```

**3. Warp divergence**
```cpp
// Bad - threads in same warp take different paths
if (threadIdx.x % 2 == 0) {
    // Half warp does this
} else {
    // Half warp does this
}

// Good - divergence at warp boundaries
if (threadIdx.x < 32) {
    // First warp does this
} else {
    // Second warp does this
}
```

**4. Kernel launch overhead**
```cpp
// Bad - many small kernels
for (int i = 0; i < 1000; i++) {
    small_kernel<<<1, 32>>>(data + i);
}

// Good - one large kernel
large_kernel<<<1000, 32>>>(data);

// Or use CUDA graphs for repeated patterns
```

### Training Pitfalls

**1. Learning rate too high/low**
```
Symptoms of LR too high: Loss spikes, NaN values
Symptoms of LR too low: Very slow convergence, stuck in local minima

Solution: Use learning rate finder or start with 1e-4 for Adam
```

**2. Forgetting model.train()/model.eval()**
```python
# These affect BatchNorm and Dropout behavior
model.train()  # Before training loop
model.eval()   # Before validation/inference
```

**3. Gradient accumulation bugs**
```python
# Bad - gradients accumulate incorrectly
for batch in loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Using accumulated gradients!

# Good - zero gradients first
for batch in loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()

# With gradient accumulation
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**4. Data leakage in preprocessing**
```python
# Bad - fit scaler on all data
scaler.fit(all_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)  # Leakage!

# Good - fit only on training data
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)
```

## Architecture Decision Guide

### When to Use Each Framework

| Scenario | Recommendation |
|----------|----------------|
| Research/prototyping | PyTorch |
| Production with TF ecosystem | TensorFlow |
| Classical ML | scikit-learn |
| NLP/LLM work | PyTorch + Transformers |
| On-device inference | TensorFlow Lite / ONNX |
| Maximum inference speed | TensorRT / vLLM |

### When to Use Each Training Strategy

| Scenario | Strategy |
|----------|----------|
| Single GPU, model fits | Standard training |
| Single GPU, model doesn't fit | Gradient checkpointing + mixed precision |
| Multi-GPU, model fits per GPU | DataParallel / DDP |
| Multi-GPU, model doesn't fit | FSDP / DeepSpeed ZeRO |
| Huge model (100B+) | Tensor + Pipeline parallelism |

### When to Use Each PEFT Method

| Scenario | Method |
|----------|--------|
| General fine-tuning | LoRA (r=8-32) |
| Memory-constrained | QLoRA (4-bit) |
| Task-specific heads | Adapter layers |
| Prompt-based learning | Prefix tuning |
| Very limited data | Few-shot prompting |

### When to Quantize

| Scenario | Quantization |
|----------|--------------|
| Training | Keep FP32/BF16 |
| Inference, accuracy critical | FP16/BF16 |
| Inference, speed critical | INT8 |
| Edge/mobile deployment | INT4 / GPTQ / AWQ |

## Debugging Workflows

### NaN/Inf in Training
```python
# 1. Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# 2. Check for problematic operations
# - Division by zero
# - Log of negative numbers
# - Exploding gradients

# 3. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Reduce learning rate
# 5. Check data for NaN/Inf values
assert not torch.isnan(input).any()
assert not torch.isinf(input).any()
```

### Out of Memory (OOM)
```python
# 1. Reduce batch size
# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Use mixed precision
scaler = torch.cuda.amp.GradScaler()

# 4. Clear cache between batches
torch.cuda.empty_cache()

# 5. Check for memory leaks
# - Tensors accumulating in list
# - Forgetting to detach tensors
# - Storing loss tensors instead of .item()
```

### Slow Training
```python
# 1. Profile to find bottleneck
# 2. Check DataLoader
#    - num_workers > 0?
#    - pin_memory=True?
#    - Is I/O the bottleneck?

# 3. Check for CPU-GPU sync points
#    - .item(), .numpy(), print(tensor)
#    - These force synchronization

# 4. Enable cudNN autotuner
torch.backends.cudnn.benchmark = True
```

### Model Not Learning
```
1. Verify data pipeline
   - Are labels correct?
   - Is preprocessing correct?
   - Visualize samples

2. Check loss function
   - Correct for task?
   - Reduction correct (mean vs sum)?

3. Verify gradients flow
   - Are any layers frozen accidentally?
   - Check gradient magnitudes

4. Simplify
   - Can model overfit on 1 batch?
   - Start with smaller model
```

## Memory Estimation Formulas

### Model Memory
```
Parameters memory = num_params × bytes_per_param
Gradients memory = num_params × bytes_per_param
Optimizer states (Adam) = num_params × 8 bytes (momentum + variance)

Total training ≈ params × (1 + 1 + 2) × bytes = 4× model size (FP32)
                 ≈ params × (2 + 2 + 8) = ~2× model size + 8B (mixed precision)
```

### Activation Memory
```
Per layer ≈ batch_size × seq_len × hidden_size × bytes
Transformer ≈ batch × seq × hidden × num_layers × 12 (approx)

With gradient checkpointing: √(num_layers) × single layer
```

### Quick Estimates
```
7B model FP16:  ~14GB weights, ~28GB inference, ~112GB training
13B model FP16: ~26GB weights, ~52GB inference, ~208GB training
70B model FP16: ~140GB weights, ~280GB inference, ~1.1TB training

With QLoRA (4-bit): ~4GB for 7B, ~8GB for 13B
```

## Key Hyperparameter Ranges

### Learning Rates
```
SGD: 0.01 - 0.1
Adam: 1e-4 - 1e-3
AdamW: 1e-5 - 1e-4 (for fine-tuning)
LoRA fine-tuning: 1e-4 - 3e-4
```

### Batch Sizes
```
Vision: 32-256 (larger often better)
NLP: 8-32 per GPU
LLM fine-tuning: 1-4 with gradient accumulation
```

### Regularization
```
Dropout: 0.1-0.5 (0.1 for transformers)
Weight decay: 0.01-0.1
Label smoothing: 0.1 for classification
```

### LoRA Hyperparameters
```
r (rank): 4-64 (start with 8)
lora_alpha: 16-32 (often 2×r)
lora_dropout: 0.05-0.1
target_modules: ["q_proj", "v_proj"] minimum
                ["q_proj", "k_proj", "v_proj", "o_proj"] better
```

## Grep Patterns for Quick Lookup

```bash
# Find specific topics in this file
grep -n "OOM" references/expert-knowledge.md
grep -n "pitfall" references/expert-knowledge.md
grep -n "memory" references/expert-knowledge.md
grep -n "debug" references/expert-knowledge.md
grep -n "When to" references/expert-knowledge.md
```
