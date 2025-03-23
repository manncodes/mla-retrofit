"""
Model components for Multi-head Latent Attention (MLA).
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache


class MLAAttention(nn.Module):
    """
    General implementation of Multi-head Latent Attention (MLA).
    
    This module implements MLA as described in the TransMLA paper. Instead of using
    standard key-value projection matrices, it projects to a smaller latent dimension
    and then expands back to the full dimension.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int = None,
        head_dim: int = None,
        dropout: float = 0.0,
        kv_dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = True,
        absorb: bool = False,
    ):
        super().__init__()
        
        # Set defaults if not provided
        num_kv_heads = num_kv_heads or (num_heads // 8)
        head_dim = head_dim or (hidden_size // num_heads)
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.original_head_dim = hidden_size // num_heads
        self.latent_dim = num_kv_heads * head_dim
        self.is_causal = is_causal
        self.absorb = absorb
        self.attention_dropout = dropout
        self.kv_dropout = kv_dropout
        
        # If absorb is True, we absorb the up-projection matrices into q_proj and o_proj
        if not self.absorb:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.k_up_proj = nn.Linear(head_dim, hidden_size, bias=False)
            self.v_up_proj = nn.Linear(head_dim, hidden_size, bias=False)
        else:
            # When absorbed, q_proj directly projects to output dimension
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        
        # The key and value projections go to the latent space
        self.k_proj = nn.Linear(hidden_size, self.latent_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.latent_dim, bias=bias)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand key-value vectors for group query attention.
        
        This function expands key-value vectors from (batch, num_kv_heads, seq_len, head_dim)
        to (batch, num_heads, seq_len, head_dim) by repeating each kv head n_rep times.
        
        Args:
            hidden_states: Tensor of key or value states [batch, num_kv_heads, seq_len, head_dim]
            n_rep: Number of repeats per kv head
            
        Returns:
            Expanded tensor of shape [batch, num_heads, seq_len, head_dim]
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        position_embeddings: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Multi-head Latent Attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_length, seq_length]
            position_ids: Position IDs [batch_size, seq_length]
            past_key_value: Cached key-value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use cache for key-value states
            layer_idx: Index of the current layer
            position_embeddings: Pre-computed position embeddings for rotary
            
        Returns:
            Tuple of (output, attention_weights, new_key_value_cache)
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Project query, key, and value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Apply dropout to key-value states (better regularization)
        key_states = F.dropout(key_states, p=self.kv_dropout, training=self.training)
        value_states = F.dropout(value_states, p=self.kv_dropout, training=self.training)
        
        # Reshape for multi-head attention
        if self.absorb:
            # For absorb mode, query goes directly to multi-head format
            query_states = F.dropout(query_states, p=self.kv_dropout, training=self.training)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # For non-absorb mode, query projects to original format, then gets transformed by k_up_proj
            query_states = query_states.view(bsz, q_len, self.num_heads, self.original_head_dim)
            k_up_weight = self.k_up_proj.weight.view(self.num_heads, self.original_head_dim, self.head_dim)
            query_states = torch.einsum("bthd,hdc->bhtc", query_states, k_up_weight)
        
        # Reshape key and value states
        key_states = key_states.view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply positional embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Apply rotary positional embeddings to query and key states
            # Implementation depends on model type (e.g., RoPE)
            # This is a placeholder for model-specific positional embedding application
            pass
        
        # Update cache if provided
        if past_key_value is not None and use_cache:
            # Update cache with current key-value states
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when using cache")
                
            cache_kwargs = {}
            if position_embeddings is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                
            key_states, value_states = past_key_value.update(
                key_states, value_states, layer_idx, cache_kwargs
            )
        
        # Group query attention: repeat key states to match number of query heads
        key_states = self._repeat_kv(key_states, self.num_heads // self.num_kv_heads)
        
        # For non-absorb mode, we need to expand value states using v_up_proj
        if not self.absorb:
            v_up_weight = self.v_up_proj.weight.view(
                self.num_kv_heads, 
                self.num_heads // self.num_kv_heads, 
                self.original_head_dim, 
                self.head_dim
            )
            value_states = torch.einsum("bhtc,hgdc->bhgtd", value_states, v_up_weight)
            value_states = value_states.reshape(bsz, self.num_heads, -1, self.original_head_dim)
        else:
            # For absorb mode, simply repeat values (like keys)
            value_states = self._repeat_kv(value_states, self.num_heads // self.num_kv_heads)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention weights to value states
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        # Apply output projection
        attn_output = self.o_proj(attn_output)
        
        # Return outputs based on parameters
        if not output_attentions:
            attn_weights = None
            
        if not use_cache:
            past_key_value = None
            
        return attn_output, attn_weights, past_key_value


class MLASelfAttention(MLAAttention):
    """
    A wrapper around MLAAttention that implements the self-attention pattern.
    
    This class is primarily for easy integration with existing model architectures
    that expect a self-attention module with a specific interface.
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for self-attention with MLA.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_length, seq_length]
            position_ids: Position IDs [batch_size, seq_length]
            past_key_value: Cached key-value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use cache for key-value states
            kwargs: Additional arguments passed to the MLAAttention forward method
            
        Returns:
            Tuple of (output, attention_weights, new_key_value_cache)
        """
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
