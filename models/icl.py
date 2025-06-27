import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_util import TransformerBlock, MLP, init_weights
from .icl_util import ICLBlock

class ICLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        
        self.x_1 = nn.Parameter(torch.randn(1, 1, config.d_embed))
        
        config.block_list = ['t', 'i'] * (config.n_blocks // 2) 
        
        self.blocks = nn.ModuleList([TransformerBlock(config) if sym == 't' else ICLBlock(config) for sym in config.block_list])
                    
        self.ln_out = nn.LayerNorm(config.d_embed)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        embeddings = self.embedding(x) # (B, S, E)
        
        B, S, E = embeddings.shape
        device = embeddings.device
        
        x_1 = self.x_1.expand(B, -1, -1) # (B, 1, E)
        
        icl_covariates = torch.cat([x_1, embeddings], dim=1) # (B, S+1, E)
        
        y_NP1 = torch.zeros(B, 1, E, device=device) # (B, 1, E)    
        icl_targets = torch.cat([embeddings, y_NP1], dim=1) # (B, S+1, E)
        icl_functional_update = torch.zeros(B, S+1, E, device=device) # (B, S+1, E)
            
        
        for block, sym in zip(self.blocks, self.config.block_list):
            
            if sym == 't':
                icl_covariates = block(icl_covariates)
            else:
                icl_covariates, icl_targets, icl_functional_update = block(icl_covariates, icl_targets, icl_functional_update)
            
        x = icl_functional_update[:, :-1, :] # (B, S, E)

        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)

        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        
        return logits, loss