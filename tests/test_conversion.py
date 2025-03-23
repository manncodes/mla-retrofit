"""
Tests for the MLA conversion functionality.
"""

import unittest
import torch
import torch.nn as nn
from mla_retrofit.model import MLAAttention
from mla_retrofit.utils import (
    repeat_kv,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_rotate_half,
    repeat_apply_rotary_pos_emb,
)


class TestMLAConversion(unittest.TestCase):
    """
    Test cases for MLA conversion utilities and model components.
    """
    
    def setUp(self):
        # Set up common test parameters
        self.hidden_size = 512
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 64
        self.batch_size = 2
        self.seq_len = 16
        
        # Create sample inputs
        self.x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    
    def test_mla_attention_forward(self):
        """Test forward pass of MLAAttention."""
        # Create attention module
        attn = MLAAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        
        # Create attention mask
        mask = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len)
        mask = mask.masked_fill(
            torch.arange(self.seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(0) >= 
            torch.arange(self.seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(-1),
            float("-inf"),
        )
        
        # Run forward pass
        output, _, _ = attn(self.x, attention_mask=mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
    
    def test_mla_attention_absorbed(self):
        """Test forward pass of MLAAttention with absorption."""
        # Create attention module with absorption
        attn = MLAAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            absorb=True,
        )
        
        # Run forward pass without mask (simpler test)
        output, _, _ = attn(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
    
    def test_repeat_kv(self):
        """Test the repeat_kv function."""
        # Create a sample key/value tensor
        kv = torch.randn(self.batch_size, self.num_kv_heads, self.seq_len, self.head_dim)
        
        # Repeat the KV heads
        repeated_kv = repeat_kv(kv, self.num_heads // self.num_kv_heads)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.assertEqual(repeated_kv.shape, expected_shape)
        
        # Check that heads are properly repeated
        repeat_factor = self.num_heads // self.num_kv_heads
        for i in range(self.num_kv_heads):
            for j in range(repeat_factor):
                head_idx = i * repeat_factor + j
                torch.testing.assert_close(
                    repeated_kv[:, head_idx, :, :],
                    kv[:, i, :, :]
                )
    
    def test_rotate_half(self):
        """Test the rotate_half function."""
        # Create a sample tensor
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        
        # Apply rotate_half
        rotated_x = rotate_half(x)
        
        # Check output shape
        self.assertEqual(rotated_x.shape, x.shape)
        
        # Check first half is -second half of original, and second half is first half of original
        half_dim = self.head_dim // 2
        torch.testing.assert_close(
            rotated_x[..., :half_dim],
            -x[..., half_dim:]
        )
        torch.testing.assert_close(
            rotated_x[..., half_dim:],
            x[..., :half_dim]
        )
    
    def test_apply_rotary_pos_emb(self):
        """Test apply_rotary_pos_emb function."""
        # Create sample query and key tensors
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_kv_heads, self.seq_len, self.head_dim)
        
        # Create sample cos and sin for RoPE
        cos = torch.randn(self.batch_size, self.seq_len, self.head_dim // 2)
        sin = torch.randn(self.batch_size, self.seq_len, self.head_dim // 2)
        
        # Apply rotary embeddings
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Check output shapes
        self.assertEqual(q_embed.shape, q.shape)
        self.assertEqual(k_embed.shape, k.shape)
    
    def test_repeat_rotate_half(self):
        """Test repeat_rotate_half function."""
        # Create a sample tensor
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim * 2)
        
        # Apply repeat_rotate_half
        rotated_x = repeat_rotate_half(x, 2)
        
        # Check output shape
        self.assertEqual(rotated_x.shape, x.shape)
    
    def test_repeat_apply_rotary_pos_emb(self):
        """Test repeat_apply_rotary_pos_emb function."""
        # Create sample query and key tensors with doubled head_dim for repeat
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim * 2)
        k = torch.randn(self.batch_size, self.num_kv_heads, self.seq_len, self.head_dim * 2)
        
        # Create sample cos and sin for RoPE
        cos = torch.randn(self.batch_size, self.seq_len, self.head_dim // 2)
        sin = torch.randn(self.batch_size, self.seq_len, self.head_dim // 2)
        
        # Apply rotary embeddings with repeat
        q_embed, k_embed = repeat_apply_rotary_pos_emb(q, k, cos, sin)
        
        # Check output shapes
        self.assertEqual(q_embed.shape, q.shape)
        self.assertEqual(k_embed.shape, k.shape)


if __name__ == "__main__":
    unittest.main()
