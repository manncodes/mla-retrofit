"""
Example script for fine-tuning a model converted with MLA.
"""

import os
import logging
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to demonstrate fine-tuning with a converted MLA model.
    """
    # Parameters
    model_path = "./qwen-7b-mla"  # Path to the converted model
    output_dir = "./qwen-7b-mla-finetuned"
    dataset_name = "tatsu-lab/alpaca"  # Example dataset, replace with your own
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Make sure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # For this example, we're using a simple format for instruction tuning
    def format_instruction(example):
        """Format the example into an instruction format."""
        return {
            "text": f"### Instruction: {example['instruction']}\n\n### Input: {example['input']}\n\n### Response: {example['output']}"
        }
    
    # Apply formatting and tokenization
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            [format_instruction(ex)["text"] for ex in examples],
            truncation=True,
            max_length=512,
            padding="max_length",
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    # Create Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=1000,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        fp16=True,
        remove_unused_columns=False,
        report_to="tensorboard",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving fine-tuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
