import json

for icl_mode in ["exact", "a", "b", "c"]:
    for icl_layers in [1, 2]:
        for transformer_layers in [0, 1]:
            
            config = {
                "model_type": "icl",
                "d_embed": 512,
                "d_hidden": 512,
                "max_seq_len": 128,
                "n_heads": 8,
                "n_icl_blocks": icl_layers,
                "n_transformer_blocks": transformer_layers,
                "icl_mode": icl_mode
            }

            filename = f"{icl_mode}_{icl_layers}i_{transformer_layers}t.json"

            with open(filename, "w") as f:
                json.dump(config, f, indent=4)