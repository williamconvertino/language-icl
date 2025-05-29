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
        
        self.x_1 = nn.Parameter(torch.randn(1, 1, config.d_embed)) # (B, S, E)
        
        self.feature_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_feature_blocks)])
        self.icl_blocks = nn.ModuleList([ICLBlock(config, self.embedding) for _ in range(config.n_icl_blocks)])
        self.extrapolation_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_extrapolation_blocks)])

        if config.freeze_feature_blocks:
            for block in self.feature_blocks:
                for param in block.parameters():
                    param.requires_grad = False
                    
        if config.freeze_icl_blocks:
            for block in self.icl_blocks:
                for param in block.parameters():
                    param.requires_grad = False
        
        if config.freeze_extrapolation_blocks:
            for block in self.feature_blocks:
                for param in block.parameters():
                    param.requires_grad = False
                    
        self.ln_out = nn.LayerNorm(config.d_embed)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x, targets=None, ignore_index=-1):
        
        embeddings = self.embedding(x) # (B, S, E)
        
        B, S, E = embeddings.shape
        device = embeddings.device
        
        x_1 = self.x_1.expand(B, -1, -1) # (B, 1, E)
        print(embeddings.shape)
        print(x_1.shape)
        icl_covariates = torch.cat([x_1, embeddings], dim=1) # (B, S+1, E)
        
        for block in self.feature_blocks:
            icl_covariates = block(icl_covariates) # (B, S+1, E)
        
        if self.config.n_icl_blocks == 0:
            x = icl_covariates[:, 1:, :] # (B, S, E)
        else:
            y_NP1 = torch.zeros(B, 1, E, device=device) # (B, 1, E)
            
            icl_targets = torch.cat([embeddings, y_NP1], dim=1) # (B, S+1, E)
            
            if self.config.use_identity_a0:
                icl_functional_update = icl_covariates.clone() # (B, S+1, E) Can reuse update as our initial prediction
            else:
                icl_functional_update = torch.zeros(B, S+1, E, device=device) # (B, S+1, E)
            
            for block in self.icl_blocks:
                icl_covariates, icl_targets, icl_functional_update = block(icl_covariates, icl_targets, icl_functional_update)

            x = icl_functional_update[:, :-1, :] # (B, S, E)
        
        for block in self.config.extrapolation_blocks:
            x = block(x)

        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)

        if targets is None:
            return logits, None
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=ignore_index)
        
        return logits, loss