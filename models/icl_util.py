import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .transformer_util import Attention

class VShiftAttention(nn.Module):
    def __init__(self, config, d_q=None, d_k=None, d_v=None, d_out=None):
        super().__init__()
        
        if d_q is None:
            d_q = config.d_embed
        if d_k is None:
            d_k = config.d_embed
        if d_v is None:
            d_v = config.d_embed
        if d_out is None:
            d_out = config.d_embed
        
        self.config = config
        
        self.W_q = nn.Linear(d_q, config.n_heads * config.d_embed, bias=False)
        self.W_k = nn.Linear(d_k, config.n_heads * config.d_embed, bias=False)
        self.W_v = nn.Linear(d_v, config.n_heads * config.d_embed, bias=False)
        self.W_o = nn.Linear(config.n_heads * config.d_embed, d_out, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_embed)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_embed)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k=None, v=None):
        
        if k is None:
            k = q
        if v is None:
            v = q
        
        B, S, E = q.shape
        device = q.device
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2) # (B, H, S, E)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2) # (B, H, S, E)
        v = self.W_v(v).view(B, S, self.config.n_heads, self.config.d_embed).transpose(1, 2) # (B, H, S, E)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(device) # (S, S)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale # (B, H, S, S)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1) # (B, H, S, S)
        
        diag_mask = torch.eye(S, dtype=torch.bool, device=device)
        attn_probs = attn_probs.masked_fill(diag_mask, 0) # Remove attention between a position and itself (but dont change softmax calculation)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_embed * self.config.n_heads)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output

class ICLBlock(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        
        self.config = config
        self.embedding = embedding
        
        if config.use_shift:
            self.attn = VShiftAttention(config)
        else:
            self.attn = Attention(config)
        
    def calculate_embedding_expectation(self, functional_update): 
        with torch.no_grad():
            embedding_matrix = self.embedding.weight # (V, E)
            weighted_expectation = F.softmax(functional_update @ embedding_matrix.transpose(0, 1), dim=-1) @ embedding_matrix # (B, S, E)
            return weighted_expectation.detach()
            
    def forward(self, covariates, targets, functional_update):
        
        targets = targets # (B, S, E)

        v = targets - self.calculate_embedding_expectation(functional_update) # (B, S, E)
        q = k = covariates # (B, S, E)

        delta_f = self.attn(q, k, v) # (B, S, E)

        functional_update = functional_update + delta_f # (B, S, E)

        return covariates, targets, functional_update