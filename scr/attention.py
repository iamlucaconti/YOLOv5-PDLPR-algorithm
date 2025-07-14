import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: (B, d_embed, H, W) 
        b, c, h, w = x.shape
        # (B, d_embed, H, W) -> (B, d_embed, H * W)
        x = x.view((b, c, h * w))
        # (B, d_embed, H * W) -> (B, H * W, d_embed)
        x = x.transpose(-1, -2)

        # (B, H*W, Dim)
        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape 

        # (B, H*W, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (B, H*W, Dim) -> (B, H*W, Dim * 3) -> 3 tensor of shape (B, H*W, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (B, H*W, Dim) -> (B, H*W, Dim / H) -> (B, H*W, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (B, H*W, Dim / H) @ (B, Dim / H, H*W) -> (B, H*W, H*W)
        attn_scores = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(attn_scores, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            attn_scores.masked_fill_(mask, float('-inf')) 
        
        # Divide by d_k (Dim / H). 
        # (B, H*W, H*W) -> (B, H*W, H*W)
        attn_scores /= math.sqrt(self.d_head) 
        attn_scores = F.softmax(attn_scores, dim=-1) 

        # (B, H*W, H*W) @ (B, H*W, Dim / H) -> (B, H*W, Dim / H)
        output = attn_scores @ v

        # (B, H*W, Dim / H) -> (B, H*W, Dim / H)
        output = output.transpose(1, 2) 

        # (B, H*W, Dim / H) -> (B, H*W, Dim)
        output = output.reshape(input_shape) 

        # (B, H*W, Dim) -> (B, H*W, Dim)
        output = self.out_proj(output) 
        
        # (B, H*W, Dim)

        # (B, H * W, d_embed) ->  (B, C, H * d_embed) 
        output = output.transpose(-1, -2)
        # (B, d_embed, H * W)  -> (B, d_embed, H , W) 
        output = output.view((b, c, h, w))
        # (B, d_embed, H , W)
        return output
    

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent):(B, C, H, W)
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        x = x.view((b, c, h * w)).transpose(-1, -2)
        # y (context): = (B, 512, 1, 3)
        # y = y.squeeze(2) # NOTE: Prima avevo messo questo ma va bene solo se la terza dimensione è 1
        y = y.view(y.size(0), y.size(1), -1)

        # x (latent): # (B, H*W_Q, Dim_Q)
        # y (context): # (B, H*W_KV, Dim_KV)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # (B, H*W_Q, Dim_Q) -> (B, H*W_Q, Dim_Q)
        q = self.q_proj(x)
        # (B, H*W_KV, Dim_KV) -> (B, H*W_KV, Dim_Q)
        k = self.k_proj(y)
        # (B, H*W_KV, Dim_KV) -> (B, H*W_KV, Dim_Q)
        v = self.v_proj(y)

        # (B, H*W_Q, Dim_Q) -> (B, H*W_Q, Dim_Q / H) -> (B, H*W_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (B, H*W_KV, Dim_Q) -> (B, H*W_KV, Dim_Q / H) -> (B, H*W_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (B, H*W_KV, Dim_Q) -> (B, H*W_KV, Dim_Q / H) -> (B, H*W_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (B, H, H*W_Q, Dim_Q / H) @ (B, H, Dim_Q / H, H*W_KV) -> (B, H, H*W_Q, H*W_KV)
        attn_scores = q @ k.transpose(-1, -2)
        
        # (B, H, H*W_Q, H*W_KV)
        attn_scores /= math.sqrt(self.d_head)
        
        # (B, H, H*W_Q, H*W_KV)
        attn_scores = F.softmax(attn_scores, dim=-1)
        
        # (B, H, H*W_Q, H*W_KV) @ (B, H, H*W_KV, Dim_Q / H) -> (B, H, H*W_Q, Dim_Q / H)
        output = attn_scores @ v
        
        # (B, H, H*W_Q, Dim_Q / H) -> (B, H*W_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (B, H*W_Q, H, Dim_Q / H) -> (B, H*W_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (B, H*W_Q, Dim_Q) -> (B, H*W_Q, Dim_Q)
        output = self.out_proj(output)

        # (B, H*W_Q, Dim_Q)

        # (B, H * W, d_embed) ->  (B, C, H * d_embed) -> (B, d_embed, H , W) 
        output = output.transpose(-1, -2).view((b, d_embed, h, w))
        return output