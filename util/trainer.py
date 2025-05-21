import os
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

from .device import get_device

class Trainer:
    
    learning_rate = 1e-4
    weight_decay = 1e-2
    betas = (0.9, 0.999)
    grad_clip = 1.0
    max_epochs = 20
    
    def __init__(self, model, splits, tokenizer, checkpoint=None, device_id=None):
    
        self.model = model
        self.splits = splits
        self.tokenizer = tokenizer
        self.device = get_device(device_id)
        
        self.checkpoint = checkpoint

        self.num_training_steps = len(self.splits["train"])
        self.num_warmup_steps = int(self.num_training_steps * 0.1)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

        self.checkpoint_dir = f"checkpoints/{self.model.config.name}"
        self.log_dir = f"logs/{self.model.config.name}"
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{dt}.log")

    def _save_checkpoint(self, epoch, val_loss, best_val_loss):
        
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_state_dict": self.model.state_dict()
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt"))
        
        if val_loss <= best_val_loss:
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": val_loss
            }
            torch.save(best_checkpoint, os.path.join(self.checkpoint_dir, "best.pt"))

    def _log(self, epoch, train_loss, val_loss):
        with open(self.log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/{self.max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n")

    def _step(self, batch):
        input_ids = batch.to(self.device)
        x = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        _, loss = self.model(x, targets=targets, ignore_index=self.tokenizer.pad_token_id)
        return loss

    def _validate(self):
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for batch in self.splits["val"]:
                loss += self._step(batch).item()
        self.model.train()
        return loss / len(self.splits["val"])

    def train(self):
        
        self.model.to(self.device)
        self.model.train()
        
        best_val_loss = float("inf") if self.checkpoint is None else self.checkpoint["best_val_loss"]
        starting_epoch = 0 if self.checkpoint is None else self.checkpoint["epoch"] + 1
        
        if self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])
            self.model.load_state_dict(self.checkpoint["model_state_dict"])

        for epoch in range(starting_epoch, self.max_epochs):
            
            batch_tqdm = tqdm(self.splits["train"], desc=f"Epoch {epoch + 1}/{self.max_epochs}", leave=False)
            total_loss = 0.0
            
            for batch in batch_tqdm:
                
                batch = batch.to(self.device)
                
                loss = self._step(batch)
        
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss = loss.item()
                total_loss += train_loss
                
                batch_tqdm.set_postfix(loss=train_loss)
                
            val_loss = self._validate()
            avg_train_loss = total_loss / len(self.splits["train"])
            
            best_val_loss = min(best_val_loss, val_loss)
            self._save_checkpoint(epoch, val_loss, best_val_loss)
            
            print(f"Epoch {epoch + 1}/{self.max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            self._log(epoch, avg_train_loss, val_loss)