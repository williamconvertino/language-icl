import util.cache

from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from models.transformer_model import TransformerModel
from models.icl_model import ICLModel
from util.loading import load_checkpoint, load_config

config = load_config("icl_shiftx_mlpx_1l")

tokenizer = Tokenizer()
config.vocab_size = tokenizer.vocab_size

model = ICLModel(config)

print(model)