import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_util import TransformerBlock, init_weights

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        if config.random_features:
            for block in self.transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = False
        
        self.ln_out = nn.LayerNorm(config.d_embed)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight
        
        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape
        device = x.device
        
        x = self.embedding(x) # (B, S, E)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x) # (B, S, E)

        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)
        
        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss