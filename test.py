from models.icl_model import ICLModel
from util.loading import load_checkpoint, load_config
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset

config = load_config("exact_1i_0t")
tokenizer = Tokenizer()
config.vocab_size = tokenizer.vocab_size

splits = TinyStoriesDataset.get_splits(tokenizer, config.max_seq_len)

model = ICLModel(config)

for batch in splits["train"]:
    model(batch)
    break