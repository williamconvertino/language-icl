import os

model_dirs = os.listdir("checkpoints")

model_epochs = []

for model_dir in model_dirs:
    checkpoint_files = os.listdir(os.path.join("checkpoints", model_dir))
    checkpoint_files = [f for f in checkpoint_files if f.endswith(".pt") and f.startswith("epoch_")]
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    latest_checkpoint = checkpoint_files[0] if checkpoint_files else None
    model_epochs.append((model_dir, int(latest_checkpoint.split('_')[1].split('.')[0]) if latest_checkpoint else None))
    # print(f"Model: {model_dir} | Epoch: { if latest_checkpoint else 'N/A'}")
    
model_epochs.sort(key=lambda x: x[1] if x is not None and x[1] is not None else float('inf'))
print("Model Checkpoints Sorted by Epochs:")
for model, epoch in model_epochs:
    print(f"Model: {model} | Epoch: {epoch if epoch is not None else 'N/A'}")