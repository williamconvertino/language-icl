import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_util import TransformerBlock, MLP, init_weights
from .icl_util import ICLBlock

class AltModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed // 2)
        
        self.x_1 = nn.Parameter(torch.randn(1, 1, config.d_embed // 2)) # (B, S, E)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.ln_out = nn.LayerNorm(config.d_embed // 2)
        self.lm_head = nn.Linear(config.d_embed // 2, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        embeddings = self.embedding(x) # (B, S, E/2)
        
        B, S, E = embeddings.shape
        device = embeddings.device
        
        x_1 = self.x_1.expand(B, -1, -1) # (B, 1, E/2)
        y_NP1 = torch.zeros(B, 1, E, device=device) # (B, 1, E/2)
        
        icl_covariates = torch.cat([x_1, embeddings], dim=1) # (B, S+1, E/2)
        icl_targets = torch.cat([embeddings, y_NP1], dim=1) # (B, S+1, E/2)

        x = torch.cat([icl_covariates, icl_targets], dim=-1) # (B, S+1, E)
        
        for block in self.blocks:
            x = block(x)

        x = x[:, 1:, self.config.d_embed // 2:]

        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)

        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        
        return logits, loss