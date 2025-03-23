# Integrating MLA-Retrofitted Models with Inference Frameworks

This guide explains how to use your MLA-retrofitted models with popular inference frameworks like vLLM, HuggingFace Transformers, and other optimized runtimes.

## HuggingFace Transformers

MLA-retrofitted models are fully compatible with standard HuggingFace Transformers inference. After conversion, you can use them just like any other model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_path = "./qwen-7b-mla"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate text
inputs = tokenizer("Tell me about Multi-head Latent Attention.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## vLLM Integration

For high-performance inference, vLLM is an excellent choice. MLA-retrofitted models, especially those converted with `rope_mode=extend` and `absorb=True`, are compatible with vLLM:

```python
from vllm import LLM, SamplingParams

# Initialize vLLM with the MLA model
model_path = "./qwen-7b-mla"
llm = LLM(model=model_path, tensor_parallel_size=1)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100
)

# Generate completions
prompts = ["Explain the benefits of Multi-head Latent Attention."]
outputs = llm.generate(prompts, sampling_params)

# Print the generated text
for output in outputs:
    print(output.outputs[0].text)
```

### vLLM Compatibility Notes

1. When using vLLM, always convert your model with `rope_mode=extend` for best compatibility.
2. The `absorb=True` option is recommended for vLLM as it simplifies the attention pattern.
3. Recent versions of vLLM (>=0.3.0) have better support for custom attention patterns.

If you encounter issues with vLLM, you can try adding custom KV cache support:

```python
# For advanced users - custom PagedAttention kernel implementation
# This would need to be integrated with vLLM's codebase
```

## ONNX Export

You can export MLA-retrofitted models to ONNX format for deployment on various inference engines:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch

# Load model and tokenizer
model_path = "./qwen-7b-mla"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create dummy inputs
dummy_input = tokenizer("This is a test", return_tensors="pt")
input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]

# Export to ONNX
output_path = Path("./onnx_model")
output_path.mkdir(exist_ok=True)

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    output_path / "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
    opset_version=15,
)

# Save tokenizer for the ONNX model
tokenizer.save_pretrained(output_path)
```

## TensorRT-LLM

For NVIDIA GPUs, TensorRT-LLM provides optimized inference. Here's how to use it with MLA-retrofitted models:

```python
# This is a simplified example - actual implementation would require more setup
import tensorrt_llm
from tensorrt_llm.models import LLaMAForCausalLM
import torch

# Load and prepare model for TensorRT-LLM
model_path = "./qwen-7b-mla"

# Build TensorRT-LLM engine
# Implementation depends on the specific model architecture
```

## Optimum for Optimized Inference

HuggingFace's Optimum library provides an easy way to accelerate inference:

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model_path = "./qwen-7b-mla"
model = ORTModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate text
inputs = tokenizer("Explain MLA in simple terms:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Tips for Optimal Performance

### Memory Optimization

MLA models already have reduced memory footprint, but you can further optimize:

1. **KV cache quantization**: Use 8-bit or 4-bit quantization for the KV cache:

   ```python
   # Example with bitsandbytes
   import bitsandbytes as bnb
   from transformers import AutoModelForCausalLM
   
   model = AutoModelForCausalLM.from_pretrained(
       "./qwen-7b-mla",
       load_in_8bit=True,  # Load model in 8-bit
       device_map="auto"
   )
   ```

2. **Gradient checkpointing**: For fine-tuning large models:

   ```python
   model.gradient_checkpointing_enable()
   ```

### Speeding Up Inference

1. **Flash Attention**: Enable Flash Attention for faster inference:

   ```python
   from transformers import AutoModelForCausalLM
   
   model = AutoModelForCausalLM.from_pretrained(
       "./qwen-7b-mla",
       device_map="auto",
       attn_implementation="flash_attention_2"
   )
   ```

2. **Batch processing**: Process multiple requests in a batch:

   ```python
   inputs = tokenizer(
       ["Question 1?", "Question 2?", "Question 3?"],
       padding=True,
       return_tensors="pt"
   ).to(model.device)
   
   outputs = model.generate(**inputs, max_new_tokens=100)
   ```

## Troubleshooting Common Issues

### Issue: Model produces incorrect or degraded outputs after conversion

Solution:
- Try converting without the `absorb` option
- Fine-tune the model after conversion
- Check if your inference framework supports the attention pattern

### Issue: Out of memory errors

Solution:
- Lower the batch size
- Use 8-bit or 4-bit quantization
- Reduce the maximum sequence length
- Use a model with fewer parameters

### Issue: Slow inference speed

Solution:
- Use Flash Attention
- Try vLLM or another optimized inference framework
- Enable tensor parallelism if using multiple GPUs
- Use the `absorb` option during conversion

### Issue: Incompatibility with specific frameworks

Solution:
- For vLLM: Use `rope_mode=extend` and `absorb=True`
- For TensorRT-LLM: Custom kernel implementations may be needed
- For ONNX: Export with dynamic axes and appropriate opset version

## Real-world Deployment Examples

### API Server with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model and tokenizer
model_path = "./qwen-7b-mla"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/generate")
async def generate(query: Query):
    try:
        inputs = tokenizer(query.prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=query.max_new_tokens,
                temperature=query.temperature,
                top_p=query.top_p,
                do_sample=query.temperature > 0,
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

By following these integration guidelines, you can effectively deploy your MLA-retrofitted models with various inference frameworks to achieve optimal performance.
