import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
    
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Parameter(torch.zeros(config.n_heads, config.d_hidden, config.d_hidden), requires_grad=True)
        self.W_k = nn.Parameter(torch.zeros(config.n_heads, config.d_hidden, config.d_hidden), requires_grad=True)
        self.W_v = nn.Parameter(torch.zeros(config.n_heads, config.d_hidden, config.d_hidden), requires_grad=True)
        self.W_o = nn.Parameter(torch.zeros(config.n_heads * config.d_hidden, config.d_hidden), requires_grad=True)

        self.attn_scale = 1 / math.sqrt(config.d_hidden)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_hidden)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k=None, v=None):
        
        if k is None:
            k = q
        if v is None:
            v = q
        
        B, S, D = q.shape # (B, S, D)
        
        q = torch.einsum('bsz,hzd->bhsd', q, self.W_q) # (B, H, S, D)
        k = torch.einsum('bsz,hzd->bhsd', k, self.W_k) # (B, H, S, D)
        v = torch.einsum('bsz,hzd->bhsd', v, self.W_v) # (B, H, S, D)
        
        q = self.rotary_embeddings(q) # (B, H, S, D)
        k = self.rotary_embeddings(k) # (B, H, S, D)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device) # (S, S)
        
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) # (B, H, S, S)
        attn_scores = attn_scores * self.attn_scale # (B, H, S, S)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf')) # (B, H, S, S)
        
        attn_probs = F.softmax(attn_scores, dim=-1) # (B, H, S, S)
        attn_probs = self.drop_attn(attn_probs) # (B, H, S, S)
        
        attn_output = torch.einsum('bhzs,bhzd->bhsd', attn_probs, v) # (B, H, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_hidden * self.config.n_heads) # (B, S, H * D)
        attn_output = torch.einsum('bsz,zd->bsd', attn_output, self.W_o) # (B, S, D)
        attn_output = self.drop_resid(attn_output) # (B, S, D)
        
        return attn_output
    
class MLP(nn.Module):
    def __init__(self, config, d_in=None, d_out=None):
        super().__init__()

        if d_in is None:
            d_in = config.d_hidden
        if d_out is None:
            d_out = config.d_hidden

        self.fc_1 = nn.Linear(d_in, 4 * config.d_hidden)
        self.fc_2 = nn.Linear(4 * config.d_hidden, d_out)
        
        self.activation = nn.GELU()    
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.attention = Attention(config)
        self.ln_attn = nn.LayerNorm(config.d_hidden)
        
        self.mlp = MLP(config)
        self.ln_mlp = nn.LayerNorm(config.d_hidden)
        
    def forward(self, x):
        x = x + self.attention(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

def init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=0.02)