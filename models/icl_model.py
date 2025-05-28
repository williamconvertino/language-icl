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
        
        self.feature_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_feature_layers)])
        self.icl_blocks = nn.ModuleList([ICLBlock(config, self.embedding) for _ in range(config.n_icl_layers)])

        if config.random_features:
            for block in self.feature_blocks:
                for param in block.parameters():
                    param.requires_grad = False
                    
        if config.freeze_icl_layers:
            for block in self.icl_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        if config.extrapolation_mode == "attn":
            self.extrapolation_block = TransformerBlock(config)
        elif config.extrapolation_mode == "mlp":
            self.extrapolation_block = MLP(config)

        self.ln_out = nn.LayerNorm(config.d_embed)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        embeddings = self.embedding(x) # (B, S, E)
        
        B, S, E = embeddings.shape
        device = x.device
        
        icl_covariates = embeddings
        icl_targets = embeddings
        icl_functional_update = torch.zeros_like(embeddings, device=device)
        
        if self.config.use_shift:
            icl_targets = icl_targets[:, 1:, :] # (B, S-1, E)
            icl_targets = torch.cat([icl_targets, torch.zeros(B, 1, E, device=device)], dim=1) # (B, S, E)
        
        for block in self.feature_blocks:
            icl_covariates = block(icl_covariates) # Update feature representations

        for block in self.icl_blocks:
            icl_covariates, icl_targets, icl_functional_update = block(icl_covariates, icl_targets, icl_functional_update)

        x = icl_functional_update
        
        if self.config.extrapolation_mode == "attn" or self.config.extrapolation_mode == "mlp":
            x = self.extrapolation_block(x)

        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)

        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        
        return logits, loss