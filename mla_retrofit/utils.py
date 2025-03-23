"""
Utility functions for MLA-Retrofit.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def get_model_config_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Extract the configuration information for a model.
    
    This function handles different model architectures and extracts the relevant
    configuration parameters for MLA conversion.
    
    Args:
        model: The model to extract configuration information from
        
    Returns:
        Dictionary containing model configuration information
    """
    config = model.config
    config_info = {}
    
    # Common parameters across most architectures
    config_info["hidden_size"] = getattr(config, "hidden_size", None)
    config_info["num_attention_heads"] = getattr(config, "num_attention_heads", None)
    config_info["num_key_value_heads"] = getattr(config, "num_key_value_heads", None)
    config_info["head_dim"] = getattr(config, "head_dim", None)
    
    # If head_dim is not in config, calculate it from hidden_size and num_attention_heads
    if config_info["head_dim"] is None and config_info["hidden_size"] is not None and config_info["num_attention_heads"] is not None:
        config_info["head_dim"] = config_info["hidden_size"] // config_info["num_attention_heads"]
    
    # Get model-specific values based on architecture
    if hasattr(config, "model_type"):
        if config.model_type in ["llama", "mistral", "mixtral"]:
            # For LLaMA, Mistral, and Mixtral models
            pass  # Using common parameters extracted above
        elif config.model_type in ["qwen2"]:
            # Qwen2 uses a slightly different naming convention
            config_info["num_key_value_heads"] = getattr(config, "num_key_value_heads", None)
        elif config.model_type in ["phi"]:
            # Phi models
            config_info["num_key_value_heads"] = getattr(config, "num_key_value_groups", None)
        elif config.model_type in ["gemma"]:
            # Gemma models
            config_info["num_key_value_heads"] = getattr(config, "num_kv_heads", None)
    
    # Verify all required parameters are present
    missing_params = [k for k, v in config_info.items() if v is None]
    if missing_params:
        logger.warning(f"Could not extract parameters: {missing_params}. These will need to be provided manually.")
    
    return config_info


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value states for group query attention.
    
    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden
    states go from (batch, num_key_value_heads, seqlen, head_dim) to 
    (batch, num_attention_heads, seqlen, head_dim).
    
    Args:
        hidden_states: Tensor of shape [batch, num_key_value_heads, seqlen, head_dim]
        n_rep: Number of times to repeat each key-value head
        
    Returns:
        Tensor of shape [batch, num_attention_heads, seqlen, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.
    
    Used for RoPE (Rotary Position Embedding).
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with rotated hidden dimensions
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to q and k tensors.
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_kv_heads, seq_len, head_dim]
        cos: Cosine component of positional embedding
        sin: Sine component of positional embedding
        position_ids: Optional position IDs (not used, kept for compatibility)
        unsqueeze_dim: Dimension along which to unsqueeze cos and sin
        
    Returns:
        Tuple of (q_embed, k_embed) with positional information applied
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_rotate_half(x: torch.Tensor, group: int) -> torch.Tensor:
    """
    Apply rotate_half but for a grouped head dimension.
    
    Args:
        x: Input tensor
        group: Number of groups
        
    Returns:
        Tensor with rotated hidden dimensions
    """
    rotate_x = []
    dh = x.shape[-1] // group
    for i in range(group):
        rotate_x.append(-x[..., i*dh + dh//2 : (i+1)*dh])
        rotate_x.append(x[..., i*dh: i*dh + dh//2])
    return torch.cat(rotate_x, dim=-1)


def repeat_apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to q and k tensors with repeated dimensions.
    
    This is a variant of apply_rotary_pos_emb that works with repeated dimensions,
    used for MLA with rope_repeat=True.
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_kv_heads, seq_len, head_dim]
        cos: Cosine component of positional embedding
        sin: Sine component of positional embedding
        position_ids: Optional position IDs (not used, kept for compatibility)
        unsqueeze_dim: Dimension along which to unsqueeze cos and sin
        
    Returns:
        Tuple of (q_embed, k_embed) with positional information applied
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    repeat_k = k.shape[-1] // cos.shape[-1]
    k_embed = (k * cos.repeat(1, 1, 1, repeat_k)) + (
        repeat_rotate_half(k, repeat_k) * sin.repeat(1, 1, 1, repeat_k)
    )
    
    repeat_q = q.shape[-1] // cos.shape[-1]
    q_embed = (q * cos.repeat(1, 1, 1, repeat_q)) + (
        repeat_rotate_half(q, repeat_q) * sin.repeat(1, 1, 1, repeat_q)
    )
    
    return q_embed, k_embed


def get_model_save_path(output_dir: str, model_name: str, suffix: str = "mla") -> Path:
    """
    Create a path for saving a converted model.
    
    Args:
        output_dir: Base output directory
        model_name: Original model name
        suffix: Suffix to add to the model name
        
    Returns:
        Path object for the save directory
    """
    # Clean model name to get just the base name
    base_name = model_name.split('/')[-1]
    save_dir = Path(output_dir) / f"{base_name}-{suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir
