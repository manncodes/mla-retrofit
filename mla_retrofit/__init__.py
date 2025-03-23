"""
MLA-Retrofit: A toolkit for retrofitting Multi-head Latent Attention (MLA) to pretrained language models.
"""

__version__ = "0.1.0"

from mla_retrofit.convert import convert_to_mla
from mla_retrofit.model import MLAAttention, MLASelfAttention

__all__ = ["convert_to_mla", "MLAAttention", "MLASelfAttention"]
