import tiktoken

class Tokenizer:

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self):
        tokenizer_base = tiktoken.get_encoding("r50k_base")
        num_base_tokens = tokenizer_base.n_vocab

        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|pad|>"
        ]

        self.special_tokens = {
            token: i + num_base_tokens for i, token in enumerate(special_tokens)
        }

        self.eos_token_id = self.special_tokens["<|end_of_text|>"]
        self.bos_token_id = self.special_tokens["<|begin_of_text|>"]
        self.pad_token_id = self.special_tokens["<|pad|>"]

        self.tokenizer = tiktoken.Encoding(
            name="tokenizer",
            pat_str=self.pat_str,
            mergeable_ranks=tokenizer_base._mergeable_ranks,
            special_tokens=self.special_tokens
        )

        self.vocab_size = self.tokenizer.n_vocab + len(self.special_tokens)
    
    def __len__(self):
        return self.vocab_size

    def _encode(self, text, eos=False, bos=False):
        sequence = []
        
        if bos:
            sequence.append(self.special_tokens["<|begin_of_text|>"])
        
        sequence.extend(self.tokenizer.encode(text, allowed_special=set(["<|begin_of_text|>", "<|end_of_text|>", "<|pad|>"])))

        if eos:
            sequence.append(self.special_tokens["<|end_of_text|>"])
        
        return sequence

    def encode(self, text, eos=False, bos=False):
        if isinstance(text, str):
            return self._encode(text, eos, bos)
        elif isinstance(text, list):
            return [self._encode(t, eos, bos) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}, expected str or list")

    def decode(self, sequence, skip_special_tokens=False):
        if len(sequence) == 0:
            return ''
        if isinstance(sequence[0], list):
            if skip_special_tokens:
                sequence = [[s for s in s_list if s not in [self.eos_token_id, self.bos_token_id, self.pad_token_id]] for s_list in sequence]
            sequence = [[s for s in s_list if s < self.tokenizer.n_vocab] for s_list in sequence]
            return [self.tokenizer.decode(s) for s in sequence]
        elif isinstance(sequence[0], int):
            if skip_special_tokens:
                sequence = [s for s in sequence if s not in [self.eos_token_id, self.bos_token_id, self.pad_token_id]]
            sequence = [s for s in sequence if s < self.tokenizer.n_vocab]
            return self.tokenizer.decode(sequence)
        else:
            raise ValueError(f"Invalid input type: {type(sequence)}, expected list of lists or list of ints")