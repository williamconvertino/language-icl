import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_util import TransformerBlock, MLP, init_weights
from .icl_util import ICLBlock

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
                    
        self.ln_out = nn.LayerNorm(config.d_embed)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        x = self.embedding(x)
        
        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)

        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        
        return logits, loss