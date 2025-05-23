import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class ICLAttentionAlt(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Parameter(torch.zeros(config.n_heads, config.d_embed, config.d_embed), requires_grad=True)
        self.W_k = nn.Parameter(torch.zeros(config.n_heads, config.d_embed, config.d_embed), requires_grad=True)
        self.W_v = nn.Parameter(torch.zeros(config.n_heads, config.d_embed, config.d_embed), requires_grad=True)
        self.W_o = nn.Parameter(torch.zeros(config.n_heads * config.d_embed, config.d_embed), requires_grad=True)

        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_embed)
        
        self.attn_scaling = 1.0 / config.d_embed

        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape
        
        q = torch.einsum('bsz,hzd->bhsd', q, self.W_q) # (B, H, S, D)
        k = torch.einsum('bsz,hzd->bhsd', k, self.W_k) # (B, H, S, D)
        v = torch.einsum('bsz,hzd->bhsd', v, self.W_v) # (B, H, S, D)
        
        q = self.rotary_embeddings(q) # (B, H, S, E)
        k = self.rotary_embeddings(k) # (B, H, S, E)

        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device) # (S, S)
        
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) # (B, H, S, S)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf')) # (B, H, S, S)
        
        attn_scores = attn_scores * self.attn_scaling

        attn_probs = F.softmax(attn_scores, dim=-1) # (B, H, S, S)
        
        # This mask avoids "cheating" by looking ahead at x_{N+1}
        # Note: we need to apply it here, rather than in causal mask, or the softmax calculations will be wrong
        diag_mask = torch.ones(S, S, device=attn_probs.device) - torch.eye(S, device=attn_probs.device) # (S, S)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0) # (1, 1, S, S)

        attn_probs = attn_probs * diag_mask # (B, H, S, S)
        
        attn_probs = self.drop_attn(attn_probs) # (B, H, S, S)

        attn_output = torch.einsum('bhzs,bhzd->bhsd', attn_probs, v) # (B, H, S, E)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_embed * self.config.n_heads) # (B, S, H * E)

        attn_output = torch.einsum('bsz,zd->bsd', attn_output, self.W_o) # (B, S, D)

        return attn_output

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

        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device) # (S, S)
        
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) # (B, H, S, S)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf')) # (B, H, S, S)
        
        attn_probs = F.softmax(attn_scores, dim=-1) # (B, H, S, S)
        
        # This mask avoids "cheating" by looking ahead at x_{N+1}
        diag_mask = torch.ones(S, S, device=attn_probs.device) - torch.eye(S, device=attn_probs.device) # (S, S)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0) # (1, 1, S, S)

        attn_probs = attn_probs * diag_mask # (B, H, S, S)
        
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
        elif self.config.icl_mode == "alt":
            self.attn = ICLAttentionAlt(config)
        
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