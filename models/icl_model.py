import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_util import TransformerBlock, init_weights
from .transformer_util import MLP
from .icl_util import ICLBlock

class ICLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.covariate_embedding = nn.Embedding(config.vocab_size, config.d_embed)

        self.icl_blocks = nn.ModuleList([ICLBlock(config, self.embedding) for _ in range(config.n_icl_blocks)])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_transformer_blocks)])
        
        if config.use_mlp:
            self.mlp = MLP(config)

        self.ln_out = nn.LayerNorm(config.d_embed)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        B, S = x.shape
        device = x.device
        
        icl_targets = self.embedding(x) # (B, S, E)
        icl_covariates = self.covariate_embedding(x) # (B, S, E)

        # Targets should be shifted versions of embeddings
        icl_targets = torch.cat((icl_targets[:, 1:, :], torch.zeros(B, 1, self.config.d_embed).to(device)), dim=1) # (B, S, E)

        # Currently initialized as zero, but may be worth changing to account for inital skip connection
        icl_functional_update = torch.zeros_like(icl_targets) # (B, S, E)

        for icl_block in self.icl_blocks:
            icl_covariates, icl_targets, icl_functional_update = icl_block(icl_covariates, icl_targets, icl_functional_update)

        x = icl_functional_update

        if self.config.use_mlp:
            x = self.mlp(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x) # (B, S, E)

        x = x[:, :, :self.config.d_embed] # (B, S, E)

        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)

        if targets is None:
            return logits, None
    
        masked_targets = targets.clone()
        masked_targets[:, 0] = ignore_index # Prevent prediction of the first token (not possible with current setup)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), masked_targets.contiguous().view(-1), ignore_index=ignore_index)
        return logits, loss