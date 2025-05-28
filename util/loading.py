import os
import torch
from types import SimpleNamespace
import json

def load_checkpoint(model, checkpoint_type=None):
    if checkpoint_type is None:
        return None
    elif checkpoint_type == "best":
        checkpoint_path = f"checkpoints/{model.config.name}/best.pt"
    elif checkpoint_type == "recent":
        checkpoint_dir = f"checkpoints/{model.config.name}"
        if not os.path.exists(checkpoint_dir):
            return None
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pt")]
        if not checkpoint_files:
            return None
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
    elif "epoch_" in checkpoint_type:
        checkpoint_path = f"checkpoints/{model.config.name}/{checkpoint_type}.pt"
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Epoch not found: {checkpoint_path}")
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
    
    if not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path, weights_only=False, map_location="cpu")

def load_config(config_name):
    
    config_dir = f"configs/{config_name}.json"
    default_config_dir = f"configs/default.json"
    
    if not os.path.exists(config_dir):
        raise ValueError(f"Config file not found: {config_dir}")
    
    def dict_to_namespace(d, default=None):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        if default:
            for k, v in default.items():
                if k not in d:
                    d[k] = v
        return SimpleNamespace(**d)
    
    # default_config = json.load(open(default_config_dir))
    default_config = None
    config = json.load(open(config_dir))
    config = dict_to_namespace(config, default_config)
    config.name = config_name
    
    if hasattr(config, "p_primary"):
        config.d_primary = int(config.d_embed * config.p_primary)
        config.d_secondary = config.d_embed - config.d_primary
        print(f"Initializing config with d_primary: {config.d_primary}, d_secondary: {config.d_secondary}")
    
    return config