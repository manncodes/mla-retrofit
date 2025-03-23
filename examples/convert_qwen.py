"""
Example script for converting Qwen2 models to use MLA.
"""

import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer

from mla_retrofit import convert_to_mla

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to demonstrate the conversion process for Qwen2.
    """
    # Model parameters
    model_name = "Qwen/Qwen2.5-7B"
    output_dir = "./qwen-7b-mla"
    num_kv_heads = 4
    head_dim = 256
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Convert model
    logger.info(f"Converting {model_name} to use MLA...")
    model, tokenizer = convert_to_mla(
        model_name_or_path=model_name,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_mode="extend",  # "extend" is more compatible with existing infrastructure
        absorb=True,  # Absorb projections for optimization
        flash_attn=False,  # Set to True if you have Flash Attention installed
        return_model=True,
    )
    
    # Save the converted model
    logger.info(f"Saving converted model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Test generation with the converted model
    logger.info("Testing generation with the converted model...")
    test_prompt = "Write a short poem about artificial intelligence."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
    
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    logger.info(f"Generated text:\n{generated_text}")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
