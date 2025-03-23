"""
Core functionality for converting models from GQA to MLA.
"""

import logging
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Tuple, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

from mla_retrofit.utils import get_model_config_info

logger = logging.getLogger(__name__)


def _init_rope_extend(
    model: nn.Module,
    hidden_size: int,
    n_heads: int,
    kv_heads: int,
    kv_dim: int,
    ori_kv_heads: int,
    ori_head_dim: int,
    latent_dim: int,
) -> None:
    """
    Initialize weights for the rope_extend mode.
    
    This mode uses identity matrices for initializing the k_up_proj and v_up_proj layers,
    and reorders them to maintain performance. This creates a model that is functionally
    equivalent to the original GQA model but with the MLA structure.
    
    Args:
        model: The model to initialize
        hidden_size: Model hidden size
        n_heads: Number of attention heads
        kv_heads: Number of key-value heads
        kv_dim: Key-value dimension per head
        ori_kv_heads: Original number of key-value heads
        ori_head_dim: Original head dimension
        latent_dim: Latent dimension (kv_heads * kv_dim)
    """
    for name, module in model.named_modules():
        if 'k_up_proj' in name or "v_up_proj" in name:
            # Create identity matrices for initializing the k_up_proj and v_up_proj layers
            weight = torch.cat([
                torch.stack([
                    torch.eye(kv_dim).reshape(-1, ori_head_dim, kv_dim)
                ] * (n_heads//ori_kv_heads), dim=1)
            ] * kv_heads).reshape(hidden_size, kv_dim).contiguous()
            
            # For k_up_proj, we need to reorder the weight
            if 'k_up_proj' in name:
                weight = weight.view(hidden_size, -1, ori_head_dim).transpose(1, 2).reshape(hidden_size, kv_dim).contiguous()
                
            # Set the weight data
            module.weight.data = weight.to(module.weight.data.device, module.weight.data.dtype)
            
        # For k_proj, we also need to reshape the weights to match the MLA structure
        elif 'k_proj' in name:
            # Reshape the weight and bias data for k_proj
            module.weight.data = module.weight.data.view(
                kv_heads, -1, ori_head_dim, hidden_size
            ).transpose(1, 2).reshape(latent_dim, hidden_size).contiguous()
            
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.view(
                    kv_heads, -1, ori_head_dim
                ).transpose(1, 2).reshape(latent_dim).contiguous()


def _init_rope_repeat(
    model: nn.Module,
    hidden_size: int,
    n_heads: int,
    kv_heads: int,
    kv_dim: int,
    ori_kv_heads: int,
    ori_head_dim: int,
    latent_dim: int,
) -> None:
    """
    Initialize weights for the rope_repeat mode.
    
    This mode simply inserts identity matrices for the k_up_proj and v_up_proj layers,
    which will be repeated.
    
    Args:
        model: The model to initialize
        hidden_size: Model hidden size
        n_heads: Number of attention heads
        kv_heads: Number of key-value heads
        kv_dim: Key-value dimension per head
        ori_kv_heads: Original number of key-value heads
        ori_head_dim: Original head dimension
        latent_dim: Latent dimension (kv_heads * kv_dim)
    """
    for name, module in model.named_modules():
        if 'k_up_proj' in name or "v_up_proj" in name:
            # Simply use identity matrices for initializing the k_up_proj and v_up_proj layers
            module.weight.data = torch.cat([
                torch.stack([
                    torch.eye(kv_dim).reshape(-1, ori_head_dim, kv_dim)
                ] * (n_heads//ori_kv_heads), dim=1)
            ] * kv_heads).reshape(hidden_size, kv_dim).contiguous().to(
                module.weight.data.device, module.weight.data.dtype
            )


def _absorb_projections(
    model: nn.Module,
    hidden_size: int,
    n_heads: int,
    kv_heads: int,
    kv_dim: int,
    ori_kv_heads: int,
    ori_head_dim: int,
    latent_dim: int,
) -> None:
    """
    Absorb the k_up_proj and v_up_proj matrices into q_proj and o_proj, respectively.
    
    This is an optimization to reduce the computational overhead of the MLA structure.
    
    Args:
        model: The model to absorb projections in
        hidden_size: Model hidden size
        n_heads: Number of attention heads
        kv_heads: Number of key-value heads
        kv_dim: Key-value dimension per head
        ori_kv_heads: Original number of key-value heads
        ori_head_dim: Original head dimension
        latent_dim: Latent dimension (kv_heads * kv_dim)
    """
    logger.info("Absorbing projections... This might take a while.")

    for name, module in model.named_modules():
        if not name.endswith("self_attn"):
            continue

        # Step 1: Absorb k_up_proj into q_proj
        k_up_weight = deepcopy(module.k_up_proj.weight.data).reshape(n_heads, ori_head_dim, kv_dim)
        q_weight = deepcopy(module.q_proj.weight.data).reshape(n_heads, ori_head_dim, hidden_size)
        
        if module.q_proj.bias is not None:
            q_weight = torch.cat([
                q_weight, deepcopy(module.q_proj.bias.data).reshape(n_heads, ori_head_dim, 1)
            ], dim=-1)
            
        # Compute the new q_proj weights by combining q_weight and k_up_weight
        q_k_up = torch.einsum("hdc,hdD->hcD", k_up_weight, q_weight)
        
        # Create a new Linear layer for q_proj
        q_proj = nn.Linear(hidden_size, n_heads*kv_dim, bias=(module.q_proj.bias is not None))
        q_proj = q_proj.to(device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype)
        
        if module.q_proj.bias is not None:
            q_proj.bias.data = q_k_up[:, :, -1].reshape(-1).contiguous()
            q_k_up = q_k_up[:, :, :-1]
            
        q_proj.weight.data = q_k_up.reshape(n_heads*kv_dim, hidden_size).contiguous()
        
        # Replace the q_proj module
        setattr(module, "q_proj", q_proj)
        
        # Remove the k_up_proj module since it's absorbed into q_proj
        delattr(module, "k_up_proj")
        
        # Step 2: Absorb v_up_proj into o_proj
        v_up_weight = deepcopy(module.v_up_proj.weight.data).reshape(n_heads, ori_head_dim, kv_dim)
        o_weight = deepcopy(module.o_proj.weight.data).reshape(hidden_size, n_heads, ori_head_dim)
        
        # Compute the new o_proj weights by combining o_weight and v_up_weight
        v_up_o = torch.einsum("hdc,Dhd->Dhc", v_up_weight, o_weight)
        
        # Create a new Linear layer for o_proj
        o_proj = nn.Linear(n_heads*kv_dim, hidden_size, bias=(module.o_proj.bias is not None))
        o_proj = o_proj.to(device=module.o_proj.weight.device, dtype=module.o_proj.weight.dtype)
        
        o_proj.weight.data = v_up_o.reshape(hidden_size, n_heads*kv_dim).contiguous()
        
        if module.o_proj.bias is not None:
            o_proj.bias.data = module.o_proj.bias.data
            
        # Replace the o_proj module
        setattr(module, "o_proj", o_proj)
        
        # Remove the v_up_proj module since it's absorbed into o_proj
        delattr(module, "v_up_proj")
        
        # Mark the module as absorbed
        module.absorb = True


def convert_to_mla(
    model_name_or_path: str,
    output_dir: Optional[str] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_mode: str = "extend",
    absorb: bool = False,
    flash_attn: bool = False,
    return_model: bool = True,
) -> Union[Tuple[PreTrainedModel, Any], None]:
    """
    Convert a GQA-based model to an MLA-based model.
    
    Args:
        model_name_or_path: The name or path of the model to convert
        output_dir: Directory to save the converted model
        num_kv_heads: Number of KV heads for MLA (required if not inferrable)
        head_dim: Head dimension for MLA (required if not inferrable)
        rope_mode: Mode for RoPE handling, either "extend" or "repeat"
        absorb: Whether to absorb projections for optimization
        flash_attn: Whether to use Flash Attention
        return_model: Whether to return the converted model and tokenizer
        
    Returns:
        Tuple of (model, tokenizer) if return_model=True, None otherwise
    """
    if rope_mode not in ["extend", "repeat"]:
        raise ValueError(f"rope_mode must be one of 'extend' or 'repeat', got {rope_mode}")
    
    # Auto-detect model architecture and load configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        attn_implementation="flash_attention_2" if flash_attn else "eager"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Extract model configuration information
    config_info = get_model_config_info(model)
    hidden_size = config_info["hidden_size"]
    n_heads = config_info["num_attention_heads"]
    
    # Determine KV heads and dimensions if not provided
    if num_kv_heads is None:
        num_kv_heads = config_info.get("num_key_value_heads", n_heads // 8)
        logger.info(f"Automatically detected num_kv_heads as {num_kv_heads}")
    
    if head_dim is None:
        head_dim = config_info.get("head_dim", hidden_size // n_heads)
        logger.info(f"Automatically detected head_dim as {head_dim}")
    
    # Calculate derived dimensions
    ori_head_dim = hidden_size // n_heads
    latent_dim = num_kv_heads * head_dim
    ori_kv_heads = latent_dim // ori_head_dim
    
    # Print configuration
    logger.info(f"Original head dimension: {ori_head_dim}")
    logger.info(f"Original KV heads: {ori_kv_heads}")
    logger.info(f"Latent dimension: {latent_dim}")
    
    # Set MLA-specific configurations
    setattr(model.config, "num_key_value_heads", num_kv_heads)
    setattr(model.config, "head_dim", head_dim)
    setattr(model.config, "rope_repeat", rope_mode == "repeat")
    
    # Initialize weights based on the selected mode
    if rope_mode == "extend":
        logger.info("Applying RoPE extend mode")
        _init_rope_extend(
            model, hidden_size, n_heads, num_kv_heads, head_dim,
            ori_kv_heads, ori_head_dim, latent_dim
        )
    elif rope_mode == "repeat":
        logger.info("Applying RoPE repeat mode")
        _init_rope_repeat(
            model, hidden_size, n_heads, num_kv_heads, head_dim,
            ori_kv_heads, ori_head_dim, latent_dim
        )
    
    # Absorb projections if specified
    if absorb:
        logger.info("Absorbing projections")
        _absorb_projections(
            model, hidden_size, n_heads, num_kv_heads, head_dim,
            ori_kv_heads, ori_head_dim, latent_dim
        )
        setattr(model.config, "absorb", True)
    
    # Save the converted model if output_dir is provided
    if output_dir:
        logger.info(f"Saving converted model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Return the model and tokenizer if requested
    if return_model:
        return model, tokenizer
    
    return None
