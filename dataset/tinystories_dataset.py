import os
from datasets import load_dataset, concatenate_datasets
from .disk_dataset import DiskDataset, DATASET_DIR

HUGGINGFACE_PATH = "roneneldan/TinyStories"

class TinyStoriesDataset(DiskDataset):
    
    def __init__(self, split, tokenizer, context_size, do_shuffle=False, allow_overlap=True):
    
        file_path = f"{DATASET_DIR}/tinystories/{split}.bin"    
        
        if not os.path.exists(file_path):

            print(f"Creating TinyStories [{split}] dataset files...")
            
            dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"{DATASET_DIR}/raw")
            dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
            
            train_test_splits = dataset.train_test_split(test_size=10000, shuffle=True, seed=42)

            train_dataset = train_test_splits["train"]
            test_dataset = train_test_splits["test"]
            
            train_val_split = train_dataset.train_test_split(test_size=10000, shuffle=True, seed=42)
            train_dataset = train_val_split["train"]
            val_dataset = train_val_split["test"]

            DiskDataset.generate_data_file(train_dataset, f"{DATASET_DIR}/tinystories/train.bin", tokenizer)
            DiskDataset.generate_data_file(test_dataset, f"{DATASET_DIR}/tinystories/test.bin", tokenizer)
            DiskDataset.generate_data_file(val_dataset, f"{DATASET_DIR}/tinystories/val.bin", tokenizer)
    
        super().__init__(file_path, tokenizer, context_size, do_shuffle=do_shuffle, allow_overlap=allow_overlap)

    def get_splits(tokenizer, max_seq_len):
        return {
            "train": TinyStoriesDataset("train", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=True),
            "val": TinyStoriesDataset("val", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=False),
            "test": TinyStoriesDataset("test", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=False)
        }