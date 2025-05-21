import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class ICLAttentionExact(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_qk_diag = nn.Parameter(torch.zeros(config.n_heads, config.d_embed), requires_grad=True) # (H, E)
        self.lr_values = nn.Parameter(torch.zeros(config.n_heads)) # (H)

        N_scaling = 1.0 / torch.arange(1, config.max_seq_len + 1).float() # (S)
        self.register_buffer("N_scaling", N_scaling.view(1, config.max_seq_len, 1)) # (1, S, 1) 

        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_embed)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape

        W_qk = torch.diag_embed(self.W_qk_diag) # (H, E, E)

        v = v.unsqueeze(1).expand(-1, self.config.n_heads, -1, -1) # (B, H, S, E)
        
        q = torch.einsum('bsz,hze->bhse', q, W_qk) # (B, H, S, E)
        k = torch.einsum('bsz,hze->bhse', k, W_qk) # (B, H, S, E)

        q = self.rotary_embeddings(q) # (B, H, S, E)
        k = self.rotary_embeddings(k) # (B, H, S, E)

        causal_mask = torch.triu(torch.ones(S, S), diagonal=0).bool().to(q.device) # (S, S)
        causal_mask[0, 0] = False # Ignore the first token (Fixes softmax)
        
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) # (B, H, S, S)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf')) # (B, H, S, S)
        
        attn_probs = F.softmax(attn_scores, dim=-1) # (B, H, S, S)
        attn_probs = self.drop_attn(attn_probs) # (B, H, S, S)
        
        attn_output = torch.einsum('bhzs,bhzd->bhsd', attn_probs, v) # (B, H, S, E)
        attn_output = attn_output * self.lr_values.view(1, self.config.n_heads, 1, 1) # (B, H, S, E)
        attn_output = attn_output.sum(dim=1) # (B, S, E)

        attn_output = attn_output * self.N_scaling[:, :S, :] # (B, S, E)

        return attn_output

class ICLBlock(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        
        self.config = config
        self.embedding = embedding

        if self.config.icl_mode == 'exact':
            self.attn = ICLAttentionExact(config)
        
    # NOTE: Only works for single layer atm (assumes functional_update is zero)
    def calculate_embedding_expectation(self, functional_update): 
        with torch.no_grad():
            embeddings = self.embedding.weight # (V, E)
            avg_embedding = embeddings.mean(dim=0).detach() # (E)
        return avg_embedding

    def forward(self, covariates, targets, functional_update):
        
        targets = targets # (B, S, E)

        v = targets - self.calculate_embedding_expectation(functional_update) # (B, S, E)
        q = k = covariates # (B, S, E)

        delta_f = self.attn(q, k, v) # (B, S, E)

        functional_update = functional_update + delta_f # (B, S, E)

        return covariates, targets, functional_update