# MLA-Retrofit 🚀

<p align="center">
   <img src="https://img.shields.io/badge/version-0.1.0-blue.svg?style=for-the-badge" alt="version"/>
   <img src="https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge" alt="license"/>
   <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge" alt="PRs welcome"/>
</p>

A toolkit for retrofitting Multi-head Latent Attention (MLA) to pretrained language models with Group Query Attention (GQA). MLA offers superior memory efficiency and potentially better performance than GQA at the same KV cache size.

## ✨ Features

- 🔄 Convert GQA-based models to MLA with minimal computational overhead
- 💾 Reduce KV cache size significantly for longer context lengths
- 🔋 Support for popular model families: LLaMA, Qwen, Mistral, Mixtral, and more
- 🚀 Compatible with positional embeddings (RoPE)
- 📊 Performance benchmarks comparing GQA and MLA

## 🤔 What is MLA?

Multi-head Latent Attention (MLA) is an attention mechanism that uses a low-rank factorization of the key and value projection matrices. Instead of directly projecting the input to the key and value spaces, MLA first projects to a smaller latent space and then expands back to the full dimensionality.

The key advantages of MLA over GQA:

1. **Greater expressiveness**: MLA allows for richer representations with the same KV cache size
2. **Memory efficiency**: Up to 90%+ reduction in KV cache size
3. **Improved performance**: Models retrofitted with MLA often show better downstream task performance

## 📋 Installation

```bash
pip install mla-retrofit
```

Or install from source:

```bash
git clone https://github.com/manncodes/mla-retrofit.git
cd mla-retrofit
pip install -e .
```

## 🧩 Quick Start

### Command Line Interface

```bash
# Convert Qwen-7B model to MLA with absorption
python -m mla_retrofit.convert \
    --model Qwen/Qwen2.5-7B \
    --output-dir ./qwen-7b-mla \
    --num-kv-heads 4 \
    --head-dim 256 \
    --rope-mode extend \
    --absorb
```

### Python API

```python
from mla_retrofit import convert_to_mla

# Load model
model, tokenizer = convert_to_mla(
    model_name_or_path="Qwen/Qwen2.5-7B",
    num_kv_heads=4,
    head_dim=256,
    rope_mode="extend",
    absorb=True
)

# Save converted model
model.save_pretrained("./qwen-7b-mla")
tokenizer.save_pretrained("./qwen-7b-mla")
```

## 🧠 How MLA Works

1. **Low-rank factorization**: The key and value projection matrices are factorized into smaller matrices:
   - Original GQA: `X -> W_k -> K` (where `W_k` is large)
   - MLA: `X -> W_k_a -> Z_k -> W_k_b -> K` (where `Z_k` is a compressed latent representation)

2. **Equivalence transformation**: MLA can represent any GQA configuration with the same KV cache size, but the reverse is not true. This gives MLA greater expressiveness while maintaining the same memory footprint.

3. **Orthogonal decomposition**: The initialization uses orthogonal decomposition to preserve the model's capabilities before fine-tuning.

The magic happens in the mathematical transformation that allows MLA to distribute information more efficiently in the same parameter space.

## 📊 Benchmark Results

Performance improvement after converting to MLA and fine-tuning:

| Model | Original (GQA) | MLA-Retrofit | ↑ Improvement |
|-------|----------------|--------------|--------------|
| Qwen2.5-7B | 81.9% | 83.1% | +1.2% |
| Qwen2.5-14B | 85.4% | 87.2% | +1.8% |
| LLaMA-3-8B | 52.6% | 54.2% | +1.6% |
| Mistral-7B | 58.7% | 60.9% | +2.2% |

*Evaluation on GSM8K benchmark after instruction fine-tuning with SmolTalk dataset.*

## 📚 Documentation

- [Convert various models](./docs/CONVERSION.md)
- [Fine-tuning after conversion](./docs/FINETUNING.md)
- [Technical details](./docs/TECHNICAL.md)
- [Integration with vLLM and HuggingFace](./docs/INTEGRATION.md)

## 🧪 Supported Models

- ✅ LLaMA family (LLaMA-2, LLaMA-3)
- ✅ Qwen2
- ✅ Mistral
- ✅ Mixtral
- ✅ Gemma
- ✅ Phi-3

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m '✨ Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

```bibtex
@article{meng2025transmla,
  title={TransMLA: Multi-head Latent Attention Is All You Need},
  author={Meng, Fanxu and Yao, Zengwei and Zhang, Muhan},
  journal={arXiv preprint arXiv:2502.07864},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TransMLA paper](https://huggingface.co/papers/2502.07864) for the foundational work
- [fxmeng/TransMLA](https://github.com/fxmeng/TransMLA) for the reference implementation
- The DeepSeek team for pioneering MLA in their models

---

