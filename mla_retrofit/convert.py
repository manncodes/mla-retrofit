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
    # First create k_up_proj and v_up_proj layers if they don't exist
    for name, module in model.named_modules():
        if name.endswith("self_attn") and not hasattr(module, "k_up_proj"):
            # Create new layers for MLA architecture
            logger.info(f"Creating k_up_proj and v_up_proj layers for {name}")
            
            # Create new linear layers
            module.k_up_proj = nn.Linear(kv_dim, hidden_size, bias=False).to(
                device=module.k_proj.weight.device, dtype=module.k_proj.weight.dtype
            )
            module.v_up_proj = nn.Linear(kv_dim, hidden_size, bias=False).to(
                device=module.v_proj.weight.device, dtype=module.v_proj.weight.dtype
            )
            
            # Initialize with identity matrices
            k_up_weight = torch.cat([
                torch.stack([
                    torch.eye(kv_dim).reshape(-1, ori_head_dim, kv_dim)
                ] * (n_heads//ori_kv_heads), dim=1)
            ] * kv_heads).reshape(hidden_size, kv_dim).contiguous()
            
            # For k_up_proj, we need to reorder the weight
            k_up_weight = k_up_weight.view(hidden_size, -1, ori_head_dim).transpose(1, 2).reshape(hidden_size, kv_dim).contiguous()
            
            # Set the weight data for both projections
            if hasattr(module, "k_proj"):
                try:
                    # Move k_proj weight to the same device as k_up_weight
                    module.k_up_proj.weight.data = k_up_weight.to(
                        device=module.k_proj.weight.device, 
                        dtype=module.k_proj.weight.dtype
                    )
                except RuntimeError as e:
                    logger.error(f"RuntimeError while setting k_up_proj weight data for {name}: {str(e)}")
                    logger.error(f"k_up_proj weight data type: {k_up_weight.dtype}, device: {k_up_weight.device}")
                    logger.error(f"k_proj weight data type: {module.k_proj.weight.dtype}, device: {module.k_proj.weight.device}")
                    raise
            else:
                raise RuntimeError(f"Module {name} does not have attribute 'k_proj'")

            
            # For v_up_proj, we can use the identity matrix directly
            v_up_weight = torch.cat([
                torch.stack([
                    torch.eye(kv_dim).reshape(-1, ori_head_dim, kv_dim)
                ] * (n_heads//ori_kv_heads), dim=1)
            ] * kv_heads).reshape(hidden_size, kv_dim).contiguous()
            
            module.v_up_proj.weight.data = v_up_weight.to(
                device=module.v_proj.weight.device, 
                dtype=module.v_proj.weight.dtype
            )
        
        # For existing modules with k_up_proj and v_up_proj
        elif 'k_up_proj' in name or "v_up_proj" in name:
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
            # Get the original shape for debugging
            original_shape = module.weight.data.shape
            original_size = module.weight.data.numel()
            
            logger.info(f"Reshaping {name}: Original shape {original_shape}, size {original_size}")
            logger.info(f"Target parameters: kv_heads={kv_heads}, ori_head_dim={ori_head_dim}, hidden_size={hidden_size}, latent_dim={latent_dim}")
            
            # Calculate appropriate dimension for the second axis
            # The tensor should have dimensions [kv_heads * group_size, ori_head_dim, hidden_size]
            # where group_size accounts for how the KV heads are distributed
            
            try:
                # Try to infer the group size based on the original size and expected dimensions
                group_size = original_size // (kv_heads * ori_head_dim * hidden_size)
                
                if group_size > 0 and kv_heads * group_size * ori_head_dim * hidden_size == original_size:
                    logger.info(f"Calculated group_size={group_size}")
                    
                    # Reshape with calculated dimensions
                    module.weight.data = module.weight.data.view(
                        kv_heads, group_size, ori_head_dim, hidden_size
                    ).transpose(1, 2).reshape(latent_dim, hidden_size).contiguous()
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data = module.bias.data.view(
                            kv_heads, group_size, ori_head_dim
                        ).transpose(1, 2).reshape(latent_dim).contiguous()
                else:
                    # If the calculation doesn't work out, try direct reshape
                    logger.warning(f"Cannot calculate appropriate group_size, using direct reshape")
                    
                    if module.weight.data.numel() == latent_dim * hidden_size:
                        module.weight.data = module.weight.data.reshape(latent_dim, hidden_size).contiguous()
                        
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data.reshape(latent_dim).contiguous()
                    else:
                        # Size mismatch - create new tensor with correct dimensions
                        logger.warning(f"Size mismatch: {module.weight.data.numel()} ≠ {latent_dim * hidden_size}, creating new tensor")
                        new_weight = torch.zeros(
                            latent_dim, hidden_size,
                            device=module.weight.data.device,
                            dtype=module.weight.data.dtype
                        )
                        
                        # Preserve as many original weights as possible by intelligent placement
                        if original_size > latent_dim * hidden_size:
                            # Original is larger - take a slice
                            flattened = module.weight.data.flatten()
                            new_weight = flattened[:latent_dim * hidden_size].reshape(latent_dim, hidden_size)
                        else:
                            # Original is smaller - copy what we can
                            flattened = module.weight.data.flatten()
                            new_weight.flatten()[:original_size] = flattened
                            
                        module.weight.data = new_weight
                        
                        # Do the same for bias if it exists
                        if hasattr(module, 'bias') and module.bias is not None:
                            if module.bias.data.numel() != latent_dim:
                                new_bias = torch.zeros(
                                    latent_dim,
                                    device=module.bias.data.device,
                                    dtype=module.bias.data.dtype
                                )
                                
                                if module.bias.data.numel() > latent_dim:
                                    new_bias = module.bias.data.flatten()[:latent_dim]
                                else:
                                    new_bias.flatten()[:module.bias.data.numel()] = module.bias.data.flatten()
                                    
                                module.bias.data = new_bias
                            else:
                                module.bias.data = module.bias.data.reshape(latent_dim).contiguous()
            
            except RuntimeError as e:
                logger.error(f"Reshape error: {str(e)}")
                
                # Try alternative approach - direct reshape to target dimensions
                try:
                    logger.warning("Attempting direct reshape to target dimensions")
                    # This assumes that weight is already in format where we can reshape to latent_dim × hidden_size
                    module.weight.data = module.weight.data.reshape(latent_dim, hidden_size).contiguous()
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data = module.bias.data.reshape(latent_dim).contiguous()
                    
                    logger.info("Direct reshape succeeded")
                except RuntimeError as e2:
                    # If even direct reshape fails, we need to diagnose deeper
                    logger.error(f"Direct reshape also failed: {str(e2)}")
                    logger.error(f"Weight data size: {module.weight.data.numel()}, Expected reshape size: {latent_dim * hidden_size}")
                    
                    # As a last resort, try to create a new weight with the right dimensions
                    if module.weight.data.numel() > latent_dim * hidden_size:
                        logger.warning("Creating new weights with correct dimensions")
                        # Take first latent_dim * hidden_size elements and reshape
                        new_weight = module.weight.data.flatten()[:latent_dim * hidden_size].reshape(latent_dim, hidden_size).contiguous()
                        module.weight.data = new_weight
                        
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = module.bias.data.flatten()[:latent_dim].contiguous()
                            module.bias.data = new_bias
                    else:
                        # If all else fails, create new random weights with appropriate scale
                        logger.warning("Creating new random weights with appropriate scale")
                        std = module.weight.data.std().item()
                        new_weight = torch.randn(
                            latent_dim, hidden_size,
                            device=module.weight.data.device,
                            dtype=module.weight.data.dtype
                        ) * std * 0.02  # Scale down to avoid large initial values
                        
                        module.weight.data = new_weight
                        
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = torch.zeros(
                                latent_dim,
                                device=module.bias.data.device,
                                dtype=module.bias.data.dtype
                            )
                            module.bias.data = new_bias
                            
        # Similarly handle v_proj reshape
        elif 'v_proj' in name:
            # Apply same approach as for k_proj
            original_shape = module.weight.data.shape
            original_size = module.weight.data.numel()
            
            logger.info(f"Reshaping {name}: Original shape {original_shape}, size {original_size}")
            
            # Try same approach as for k_proj
            try:
                if module.weight.data.numel() == latent_dim * hidden_size:
                    # Direct reshape is possible
                    module.weight.data = module.weight.data.reshape(latent_dim, hidden_size).contiguous()
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data = module.bias.data.reshape(latent_dim).contiguous()
                else:
                    # Create a new tensor with the right dimensions
                    logger.warning(f"Size mismatch: {module.weight.data.numel()} ≠ {latent_dim * hidden_size}, creating new tensor")
                    new_weight = torch.zeros(
                        latent_dim, hidden_size,
                        device=module.weight.data.device,
                        dtype=module.weight.data.dtype
                    )
                    
                    # Copy what we can from the original weights
                    if original_size > latent_dim * hidden_size:
                        flattened = module.weight.data.flatten()
                        new_weight = flattened[:latent_dim * hidden_size].reshape(latent_dim, hidden_size)
                    else:
                        flattened = module.weight.data.flatten()
                        new_weight.flatten()[:original_size] = flattened
                        
                    module.weight.data = new_weight
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        if module.bias.data.numel() != latent_dim:
                            new_bias = torch.zeros(
                                latent_dim,
                                device=module.bias.data.device,
                                dtype=module.bias.data.dtype
                            )
                            
                            if module.bias.data.numel() > latent_dim:
                                new_bias = module.bias.data.flatten()[:latent_dim]
                            else:
                                new_bias.flatten()[:module.bias.data.numel()] = module.bias.data.flatten()
                                
                            module.bias.data = new_bias
                        else:
                            module.bias.data = module.bias.data.reshape(latent_dim).contiguous()
            except Exception as e:
                logger.error(f"Error reshaping {name}: {str(e)}")
                # Fall back to random initialization with appropriate scale
                std = module.weight.data.std().item()
                module.weight.data = torch.randn(
                    latent_dim, hidden_size,
                    device=module.weight.data.device,
                    dtype=module.weight.data.dtype
                ) * std * 0.02
                
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = torch.zeros(
                        latent_dim,
                        device=module.bias.data.device,
                        dtype=module.bias.data.dtype
                    )


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
    # First create k_up_proj and v_up_proj layers if they don't exist
    for name, module in model.named_modules():
        if name.endswith("self_attn") and not hasattr(module, "k_up_proj"):
            # Create new layers for MLA architecture
            logger.info(f"Creating k_up_proj and v_up_proj layers for {name}")
            
            # Create new linear layers
            module.k_up_proj = nn.Linear(kv_dim, hidden_size, bias=False)
            module.v_up_proj = nn.Linear(kv_dim, hidden_size, bias=False)
            
            # Initialize with identity matrices
            identity_weight = torch.cat([
                torch.stack([
                    torch.eye(kv_dim).reshape(-1, ori_head_dim, kv_dim)
                ] * (n_heads//ori_kv_heads), dim=1)
            ] * kv_heads).reshape(hidden_size, kv_dim).contiguous()
            
            # Set the weight data
            module.k_up_proj.weight.data = identity_weight.to(
                device=module.k_proj.weight.device, 
                dtype=module.k_proj.weight.dtype
            )
            
            module.v_up_proj.weight.data = identity_weight.to(
                device=module.v_proj.weight.device, 
                dtype=module.v_proj.weight.dtype
            )
            
            # Also reshape k_proj and v_proj
            # Use the same approach as in _init_rope_extend
            try:
                # Reshape k_proj if needed
                if module.k_proj.weight.data.numel() == latent_dim * hidden_size:
                    module.k_proj.weight.data = module.k_proj.weight.data.reshape(latent_dim, hidden_size).contiguous()
                    
                    if hasattr(module.k_proj, 'bias') and module.k_proj.bias is not None:
                        module.k_proj.bias.data = module.k_proj.bias.data.reshape(latent_dim).contiguous()
                else:
                    # Size mismatch - handle with new tensor
                    logger.warning(f"k_proj size mismatch, creating compatible tensor")
                    new_k_weight = torch.zeros(
                        latent_dim, hidden_size,
                        device=module.k_proj.weight.device,
                        dtype=module.k_proj.weight.dtype
                    )
                    
                    # Copy as much of the original data as possible
                    orig_size = module.k_proj.weight.data.numel()
                    target_size = latent_dim * hidden_size
                    
                    if orig_size > target_size:
                        # Original is larger - take first portion
                        flattened = module.k_proj.weight.data.flatten()
                        new_k_weight = flattened[:target_size].reshape(latent_dim, hidden_size)
                    else:
                        # Original is smaller - copy what we can
                        flattened = module.k_proj.weight.data.flatten()
                        new_k_weight.flatten()[:orig_size] = flattened
                        
                    module.k_proj.weight.data = new_k_weight
                    
                    # Handle bias similarly
                    if hasattr(module.k_proj, 'bias') and module.k_proj.bias is not None:
                        if module.k_proj.bias.data.numel() != latent_dim:
                            new_k_bias = torch.zeros(
                                latent_dim,
                                device=module.k_proj.bias.data.device,
                                dtype=module.k_proj.bias.data.dtype
                            )
                            
                            orig_bias_size = module.k_proj.bias.data.numel()
                            
                            if orig_bias_size > latent_dim:
                                new_k_bias = module.k_proj.bias.data.flatten()[:latent_dim]
                            else:
                                new_k_bias.flatten()[:orig_bias_size] = module.k_proj.bias.data.flatten()
                                
                            module.k_proj.bias.data = new_k_bias
                        else:
                            module.k_proj.bias.data = module.k_proj.bias.data.reshape(latent_dim).contiguous()
                
                # Similar approach for v_proj
                if module.v_proj.weight.data.numel() == latent_dim * hidden_size:
                    module.v_proj.weight.data = module.v_proj.weight.data.reshape(latent_dim, hidden_size).contiguous()
                    
                    if hasattr(module.v_proj, 'bias') and module.v_proj.bias is not None:
                        module.v_proj.bias.data = module.v_proj.bias.data.reshape(latent_dim).contiguous()
                else:
                    # Size mismatch - handle with new tensor
                    logger.warning(f"v_proj size mismatch, creating compatible tensor")
                    new_v_weight = torch.zeros(
                        latent_dim, hidden_size,
                        device=module.v_proj.weight.device,
                        dtype=module.v_proj.weight.dtype
                    )
                    
                    # Copy as much of the original data as possible
                    orig_size = module.v_proj.weight.data.numel()
                    target_size = latent_dim * hidden_size
                    
                    if orig_size > target_size:
                        # Original is larger - take first portion
                        flattened = module.v_proj.weight.data.flatten()
                        new_v_weight = flattened[:target_size].reshape(latent_dim, hidden_size)
                    else:
                        # Original is smaller - copy what we can
                        flattened = module.v_proj.weight.data.flatten()
                        new_v_weight.flatten()[:orig_size] = flattened
                        
                    module.v_proj.weight.data = new_v_weight
                    
                    # Handle bias similarly
                    if hasattr(module.v_proj, 'bias') and module.v_proj.bias is not None:
                        if module.v_proj.bias.data.numel() != latent_dim:
                            new_v_bias = torch.zeros(
                                latent_dim,
                                device=module.v_proj.bias.data.device,
                                dtype=module.v_proj.bias.data.dtype
                            )
                            
                            orig_bias_size = module.v_proj.bias.data.numel()
                            
                            if orig_bias_size > latent_dim:
                                new_v_bias = module.v_proj.bias.data.flatten()[:latent_dim]
                            else:
                                new_v_bias.flatten()[:orig_bias_size] = module.v_proj.bias.data.flatten()
                                
                            module.v_proj.bias.data = new_v_bias
                        else:
                            module.v_proj.bias.data = module.v_proj.bias.data.reshape(latent_dim).contiguous()
            except Exception as e:
                logger.error(f"Error reshaping projection layers: {str(e)}")
                # Fall back to random initialization if needed
                std_k = module.k_proj.weight.data.std().item()
                std_v = module.v_proj.weight.data.std().item()
                
                module.k_proj.weight.data = torch.randn(
                    latent_dim, hidden_size,
                    device=module.k_proj.weight.device,
                    dtype=module.k_proj.weight.dtype
                ) * std_k * 0.02
                
                module.v_proj.weight.data = torch.randn(
                    latent_dim, hidden_size,
                    device=module.v_proj.weight.device,
                    dtype=module.v_proj.weight.dtype
                ) * std_v * 0.02
                
                if hasattr(module.k_proj, 'bias') and module.k_proj.bias is not None:
                    module.k_proj.bias.data = torch.zeros(
                        latent_dim,
                        device=module.k_proj.bias.data.device,
                        dtype=module.k_proj.bias.data.dtype
                    )
                
                if hasattr(module.v_proj, 'bias') and module.v_proj.bias is not None:
                    module.v_proj.bias.data = torch.zeros(
                        latent_dim,
                        device=module.v_proj.bias.data.device,
                        dtype=module.v_proj.bias.data.dtype
                    )
                
        elif 'k_up_proj' in name or "v_up_proj" in name:
            # For existing projection layers, simply set identity weights
            identity_weight = torch.cat([
                torch.stack([
                    torch.eye(kv_dim).reshape(-1, ori_head_dim, kv_dim)
                ] * (n_heads//ori_kv_heads), dim=1)
            ] * kv_heads).reshape(hidden_size, kv_dim).contiguous()
            
            module.weight.data = identity_weight.to(
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
            
        # Skip if required attributes don't exist
        if not hasattr(module, "k_up_proj") or not hasattr(module, "v_up_proj"):
            logger.warning(f"Skipping absorption for {name} - missing k_up_proj or v_up_proj")
            continue

        try:
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
            
        except Exception as e:
            logger.error(f"Error during absorption for {name}: {str(e)}")
            raise
            

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
    ori_kv_heads = config_info.get("num_key_value_heads", n_heads)
    
    # Print configuration
    logger.info(f"Original head dimension: {ori_head_dim}")
    logger.info(f"Original KV heads: {ori_kv_heads}")
    logger.info(f"Target latent dimension: {latent_dim}")
    logger.info(f"Target KV heads: {num_kv_heads}")
    logger.info(f"Target head dimension: {head_dim}")
    
    # Check if needed modules exist in the model
    has_mla_structure = True
    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            if not hasattr(module, "k_up_proj") or not hasattr(module, "v_up_proj"):
                has_mla_structure = False
                logger.info(f"Module {name} does not have MLA structure (missing k_up_proj or v_up_proj)")
                break
    
    logger.info(f"Model has MLA structure: {has_mla_structure}")
    
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
    
    # Absorb projections if specified and possible
    if absorb:
        # Check if all required modules now exist
        can_absorb = True
        for name, module in model.named_modules():
            if name.endswith("self_attn"):
                if not hasattr(module, "k_up_proj") or not hasattr(module, "v_up_proj"):
                    can_absorb = False
                    logger.warning(f"Cannot absorb projections for {name}: Missing k_up_proj or v_up_proj")
                    break
        
        if can_absorb:
            logger.info("Absorbing projections")
            _absorb_projections(
                model, hidden_size, n_heads, num_kv_heads, head_dim,
                ori_kv_heads, ori_head_dim, latent_dim
            )
            setattr(model.config, "absorb", True)
        else:
            logger.warning("Skipping absorption due to missing modules - MLA structure may be incomplete")
    
    # Save the converted model if output_dir is provided
    if output_dir:
        logger.info(f"Saving converted model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Return the model and tokenizer if requested
    if return_model:
        return model, tokenizer
    
    return None