# Converting Models to MLA

This guide will help you convert various model architectures from GQA to MLA.

## General Approach

The general approach for converting any model is:

1. Determine the appropriate latent dimension size
2. Choose a RoPE handling mode
3. Run the conversion
4. (Optional) Fine-tune the model

MLA-Retrofit tries to auto-detect the appropriate parameters for each model architecture, but you may need to specify them manually for optimal results.

## Command Line Usage

The simplest way to convert a model is through the command-line interface:

```bash
mla-retrofit --model MODEL_NAME_OR_PATH --output-dir OUTPUT_DIR
```

For more control, you can specify additional parameters:

```bash
mla-retrofit \
    --model MODEL_NAME_OR_PATH \
    --output-dir OUTPUT_DIR \
    --num-kv-heads NUM_KV_HEADS \
    --head-dim HEAD_DIM \
    --rope-mode extend \
    --absorb \
    --flash-attn \
    --test
```

## Model-Specific Guidelines

### LLaMA 3

```bash
mla-retrofit \
    --model meta-llama/Llama-3-8B \
    --output-dir ./llama-3-8b-mla \
    --num-kv-heads 4 \
    --head-dim 256 \
    --rope-mode extend \
    --absorb
```

### Qwen 2.5

```bash
mla-retrofit \
    --model Qwen/Qwen2.5-7B \
    --output-dir ./qwen-7b-mla \
    --num-kv-heads 4 \
    --head-dim 256 \
    --rope-mode extend \
    --absorb
```

### Mistral

```bash
mla-retrofit \
    --model mistralai/Mistral-7B-v0.1 \
    --output-dir ./mistral-7b-mla \
    --num-kv-heads 4 \
    --head-dim 128 \
    --rope-mode extend \
    --absorb
```

### Mixtral

```bash
mla-retrofit \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --output-dir ./mixtral-mla \
    --num-kv-heads 16 \
    --head-dim 128 \
    --rope-mode extend \
    --absorb
```

### Gemma

```bash
mla-retrofit \
    --model google/gemma-7b \
    --output-dir ./gemma-7b-mla \
    --num-kv-heads 4 \
    --head-dim 256 \
    --rope-mode extend \
    --absorb
```

### Phi-3

```bash
mla-retrofit \
    --model microsoft/phi-3 \
    --output-dir ./phi-3-mla \
    --num-kv-heads 16 \
    --head-dim 64 \
    --rope-mode extend \
    --absorb
```

## Parameter Selection Guide

### Choosing `num-kv-heads` and `head-dim`

The product of `num-kv-heads` and `head-dim` should equal the latent dimension size. This is typically chosen to match or be slightly smaller than the original KV cache size of the GQA model.

For example, if the original model has:
- 32 attention heads
- 4 KV heads (GQA)
- Head dimension of 128

Then the KV dimension is 4 × 128 = 512, so good MLA settings would be:
- `num-kv-heads=4`, `head-dim=128` (same as original)
- `num-kv-heads=8`, `head-dim=64` (different decomposition, same size)
- `num-kv-heads=2`, `head-dim=256` (different decomposition, same size)

### Choosing `rope-mode`

There are two options for handling rotary position embeddings:

- **extend**: More compatible with existing infrastructure, easier to integrate.
- **repeat**: Maintains exact equivalence to the original model but may require specialized inference code.

For most users, `extend` is recommended.

### Absorb Option

The `--absorb` flag combines the MLA up-projection matrices with the query and output projections. This optimization:

- Reduces computational overhead
- Makes the model more compatible with existing inference engines
- Slightly increases parameter count

It's recommended to use this option for most cases.

## Python API Usage

You can also use the Python API for more flexibility:

```python
from mla_retrofit import convert_to_mla

# Load and convert model
model, tokenizer = convert_to_mla(
    model_name_or_path="Qwen/Qwen2.5-7B",
    num_kv_heads=4,
    head_dim=256,
    rope_mode="extend",
    absorb=True,
    flash_attn=False,
    return_model=True,
)

# Save converted model
model.save_pretrained("./qwen-7b-mla")
tokenizer.save_pretrained("./qwen-7b-mla")
```

## Memory Usage Comparison

| Model | Original KV Cache | MLA KV Cache | Reduction |
|-------|------------------|--------------|-----------|
| LLaMA-3-8B | 8.0 GB | 1.0 GB | 87.5% |
| Qwen2.5-7B | 6.5 GB | 0.8 GB | 87.7% |
| Mistral-7B | 5.5 GB | 0.7 GB | 87.3% |
| Mixtral-8x7B | 44.0 GB | 5.5 GB | 87.5% |

These estimates are for 512K context length with 8-bit KV cache quantization.

## Troubleshooting

If you encounter issues during conversion:

1. **Memory errors**: Try using `--num-kv-heads` and `--head-dim` values that result in a smaller latent dimension.

2. **Generation quality**: If generation quality degrades, try:
   - Using `--rope-mode=extend` instead of `repeat`
   - Not using the `--absorb` option
   - Fine-tuning the model after conversion

3. **Compatibility with inference frameworks**: 
   - For vLLM, FlashAttention, or other frameworks, use `--rope-mode=extend`
   - If using `--absorb=True`, make sure your inference framework supports custom attention patterns

For more specific issues, please check the GitHub repository issues or create a new issue with details about your problem.
