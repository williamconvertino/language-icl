import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .transformer_util import MLP

class ICLAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.d_embed, config.n_heads * config.d_embed, bias=False)
        self.W_k = nn.Linear(config.d_embed, config.n_heads * config.d_embed, bias=False)
        self.W_v = nn.Linear(config.d_embed, config.n_heads * config.d_embed, bias=False)
        self.W_o = nn.Linear(config.n_heads * config.d_embed, config.d_embed, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_embed)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_embed)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k=None, v=None):
        
        if k is None:
            k = q
        if v is None:
            v = q
        
        B, S, E = q.shape # Note that S here really represents S+1, as we add an additional global context token
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        
        if hasattr(self.config, "use_wv") and not self.config.use_wv:
            v = v.unsqueeze(2).expand(B, S, self.config.n_heads, E).transpose(1, 2)
        else:
            v = self.W_v(v).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2)
        
        if not (hasattr(self.config, "remove_icl_rotary") and self.config.remove_icl_rotary):
            q = self.rotary_embeddings(q)
            k = self.rotary_embeddings(k)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=0).bool().to(q.device) # (S, S)
        causal_mask[0, 0] = False # (First self attention is purely global, so no harm in keeping it) (prevents softmax issues)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf')) # (B, S, S)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_embed * self.config.n_heads)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output

class ICLBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.attn = ICLAttention(config)        
        self.mlp_expectation = MLP(config)
        
    def forward(self, covariates, targets, functional_update):
        
        v = targets + self.mlp_expectation(functional_update)    
        q = k = covariates # (B, S + 1, E)

        delta_f = self.attn(q, k, v) # (B, S + 1, E)

        functional_update = functional_update + delta_f # (B, S + 1, E)

        return covariates, targets, functional_update