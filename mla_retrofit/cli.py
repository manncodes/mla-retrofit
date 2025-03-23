"""
Command-line interface for MLA-Retrofit.
"""

import argparse
import logging
import sys
from pathlib import Path

from mla_retrofit.convert import convert_to_mla
from mla_retrofit.utils import get_model_save_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a pretrained language model to use Multi-head Latent Attention (MLA)."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the model to convert",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Directory to save the converted model",
    )
    
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of key-value heads for MLA (will be auto-detected if not provided)",
    )
    
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Head dimension for MLA (will be auto-detected if not provided)",
    )
    
    parser.add_argument(
        "--rope-mode",
        type=str,
        default="extend",
        choices=["extend", "repeat"],
        help="Mode for handling rotary position embeddings",
    )
    
    parser.add_argument(
        "--absorb",
        action="store_true",
        help="Absorb projection matrices for optimization (may affect stability)",
    )
    
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use Flash Attention for faster inference",
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the model after conversion",
    )
    
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="Tell me a short story about a robot learning to be human.",
        help="Prompt to use for testing the model",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def test_model(model, tokenizer, prompt):
    """
    Quick test of the model.
    
    Args:
        model: The model to test
        tokenizer: The tokenizer to use
        prompt: The prompt to use for testing
        
    Returns:
        Generated text
    """
    # Process prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    logger.info(f"Generating text for prompt: {prompt}")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    
    # Decode and display
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    logger.info(f"Generated text: {generated_text}")
    
    return generated_text


def main():
    """Main function for the CLI."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Converting model: {args.model}")
    logger.info(f"RoPE mode: {args.rope_mode}")
    logger.info(f"Absorb projections: {args.absorb}")
    
    # Convert model
    output_dir = get_model_save_path(args.output_dir, args.model)
    model, tokenizer = convert_to_mla(
        model_name_or_path=args.model,
        output_dir=str(output_dir) if not args.test else None,  # Only save if not testing
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        rope_mode=args.rope_mode,
        absorb=args.absorb,
        flash_attn=args.flash_attn,
        return_model=True,
    )
    
    # Test model if requested
    if args.test:
        import torch
        test_model(model, tokenizer, args.test_prompt)
        
        # Save model after testing if output_dir is provided
        if args.output_dir:
            logger.info(f"Saving model to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    else:
        logger.info(f"Model saved to {output_dir}")
    
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
