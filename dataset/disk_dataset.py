import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm
import re

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/datasets")

class DiskDataset:
    
    replacements = {
        "�": "", # Unknown characters 
        "â": "",
        "€": "",
        "œ": "",
        "™": "",
        "``": '"', # Uniform quotation marks
        "''": '"',
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "..." # Ellipsis
    }
    
    re_replace = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))
    
    stride_multiplier = 0.5
    shuffle_buffer_size=1024

    def __init__(self, file_path, tokenizer, max_seq_len, do_shuffle=False, allow_overlap=True, batch_size=64):
        
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = int(max_seq_len * self.stride_multiplier)
        self.do_shuffle = do_shuffle
        self.batch_size = batch_size
        self.allow_overlap = allow_overlap # Whether to allow overlapping sequences (Used in training, not in eval)

        self.data = np.memmap(file_path, dtype="int32", mode="r")
        self.file_size = os.path.getsize(self.file_path) // np.dtype("int32").itemsize

    def __len__(self):
        num_windows = (self.file_size - self.max_seq_len) // self.stride + 1
        return math.ceil(num_windows / self.batch_size)
    
    def __iter__(self):
        read_pointer = 0
        buffer = []
        batch = []

        def pop_buffer():
            pop_index = random.randint(0, len(buffer) - 1) if self.do_shuffle else 0
            seq = buffer.pop(pop_index).copy()
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id

            if not self.allow_overlap:
                indices = np.where(seq == eos_token_id)[0]
                if indices.size > 0:
                    first_eos = indices[0]
                    seq[first_eos + 1 :] = pad_token_id
            return torch.tensor(seq)

        while read_pointer + self.max_seq_len <= len(self.data):
            chunk = self.data[read_pointer : read_pointer + self.max_seq_len].copy()
            buffer.append(chunk)
            read_pointer += self.stride

            if len(buffer) >= self.shuffle_buffer_size:
                while buffer and len(batch) < self.batch_size:
                    batch.append(pop_buffer())
                if len(batch) == self.batch_size:
                    yield torch.stack(batch).long()
                    batch = []

        while buffer:
            batch.append(pop_buffer())
            if len(batch) == self.batch_size:
                yield torch.stack(batch).long()
                batch = []
        
        if batch:
            yield torch.stack(batch).long()

    def preprocess(examples, tokenizer, separate_lines=True):
        
        # Remove/replace unwanted characters
        texts = [DiskDataset.re_replace.sub(lambda m: DiskDataset.replacements[m.group()], text) for text in examples["text"]]
        
        examples["input_ids"] = tokenizer.encode(texts, eos=separate_lines, bos=separate_lines)
        
        return examples

    def generate_data_file(dataset, file_path, tokenizer, separate_lines = True):
        
        dataset = dataset.map(lambda x: DiskDataset.preprocess(x, tokenizer, separate_lines), batched=True, remove_columns=["text"])
        file_size = sum([len(example) for example in dataset["input_ids"]])
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        memmap_array = np.memmap(file_path, dtype="int32", mode="w+", shape=(file_size,))
        
        buffer = []
        write_pointer = 0
        buffer_size=1024
        
        for sequence in tqdm(dataset["input_ids"], desc="Generating dataset files"):
            buffer.extend(sequence)
            if len(buffer) >= buffer_size:
                memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
                write_pointer += len(buffer)
                buffer = []
        
        if len(buffer) > 0:
            memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
            write_pointer += len(buffer)
            buffer = []
            
        memmap_array.flush()
        return memmap_array