import random
import torch
import torch.nn.functional as F
from .trainer import Trainer
from .device import get_device

def generate_text_nucleus(model, tokenizer, input_ids, max_length=50, temperature=1.0, top_p=0.9, device=None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    len_input_ids = len(input_ids)
    
    model.to(device)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    generated = input_ids
    
    for _ in range(max_length):
        if generated.size(1) > model.config.max_seq_len:
            input_ids = generated[:, -model.config.max_seq_len:]
        else:
            input_ids = generated
            
        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        
        probs = F.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
        
        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        generated = torch.cat((generated, next_token), dim=1).to(device)
         
    return generated[0].tolist()[len_input_ids:]

class Evaluator:
    def __init__(self, model, splits, tokenizer, checkpoint=None):
        self.model = model
        self.splits = splits
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint

        self.device = get_device()
        
    def _get_test_loss(self):
        
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in self.splits["test"]:
                input_ids = batch.to(self.device)
                x = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                _, loss = self.model(x, targets=targets, ignore_index=self.tokenizer.pad_token_id)
                test_loss += loss.item()
                
        return test_loss / len(self.splits["test"])

    def evaluate(self, num_prompts=50, do_generations=False):
        
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        
        
        if not self.model.config.name == "baseline":
            self.model.to(self.device)
            self.model.eval()

            test_loss = self._get_test_loss()
            
            print(f"=" * 50)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"=" * 50)
            
        if not do_generations:
            return
        
        prompts = []
        true_endings = []
        
        for i, batch in enumerate(self.splits["test"]):
            if i >= num_prompts:
                break
            
            example = batch[0].tolist()
            example = [token for token in example if token != self.tokenizer.pad_token_id]
            max_tokens = min(self.model.config.max_seq_len // 2, len(example) // 2)

            prompt = example[:max_tokens]      
            true_ending = example[max_tokens:]      
            
            prompts.append(prompt)
            true_endings.append(true_ending)
            
        for prompt, true_ending in zip(prompts, true_endings):
            print("=" * 50)
            print("Prompt:")
            print(self.tokenizer.decode(prompt, skip_special_tokens=True))
            print("-" * 50)
            
            if self.model.config.name == "baseline": 
                print("Baseline Text:")
                print(self.tokenizer.decode(true_ending, skip_special_tokens=True))
                print("=" * 50) 
            else:
                generated_text = generate_text_nucleus(self.model, self.tokenizer, prompt, device=self.device)
                print("Generated Text:")
                print(self.tokenizer.decode(generated_text, skip_special_tokens=True))
                print("=" * 50)