import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_util import TransformerBlock, init_weights
from .icl_util import ICLBlock

class ICLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)

        self.icl_blocks = nn.ModuleList([ICLBlock(config, self.embedding) for _ in range(config.n_icl_blocks)])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_transformer_blocks)])
        
        self.ln_out = nn.LayerNorm(config.d_embed)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape
        device = x.device
        
        x = self.embedding(x) # (B, S, E)

        icl_covariates = x[:, :, :] # (B, S, E)

        icl_targets = torch.cat((x[:, 1:, :], torch.zeros(B, 1, self.config.d_embed).to(device)), dim=1) # (B, S, E)

        icl_functional_update = torch.zeros_like(icl_targets) # (B, S, E)

        for icl_block in self.icl_blocks:
            icl_covariates, icl_targets, icl_functional_update = icl_block(icl_covariates, icl_targets, icl_functional_update)

        scratch_space = torch.zeros((B, S, self.config.d_hidden - self.config.d_embed), device=device)
        x = torch.cat([icl_functional_update, scratch_space], dim=-1) # (B, S, D)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x) # (B, S, D)

        x = x[:, :, :self.config.d_embed] # (B, S, E)

        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)

        if targets is None:
            return logits, None
    
        targets[:, 0] = ignore_index # Prevent prediction of the first token
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss