"""
Define a cross attention block, used if CA conditioning is specified
"""
import torch 
import torch.nn as nn 
#################################################################################
#                             Cross-Attention block                             #
#################################################################################
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        # Split the embedding dimension into multiple heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers to project the input x and c into query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layer for regularization
        self.attn_dropout = nn.Dropout(dropout)
        
        # Softmax to normalize attention scores
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, c):
        """
        Perform Cross-Attention.
        
        Args:
            x: Query tensor, typically the input features (batch_size, seq_len, embed_dim)
            c: Conditioning tensor, used for Key and Value (batch_size, seq_len, embed_dim)
        
        Returns:
            Output tensor after attention (batch_size, seq_len, embed_dim)
        """
        
        # Project input x and c to query, key, and value space
        q = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.key_proj(c)    # (batch_size, seq_len, embed_dim)
        v = self.value_proj(c)  # (batch_size, seq_len, embed_dim)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5  # (batch_size, num_heads, seq_len, seq_len)
        attn_probs = self.softmax(attn_scores)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply attention dropout
        attn_probs = self.attn_dropout(attn_probs)
        
        # Compute the attention output
        attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Combine heads and project back to embed_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), attn_output.size(2), self.embed_dim)
        
        # Final output projection
        output = self.out_proj(attn_output)  # (batch_size, seq_len, embed_dim)
        
        return output