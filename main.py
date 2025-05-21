import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments

from argparse import ArgumentParser
from util.trainer import Trainer
from util.evaluator import Evaluator
from util.llm_eval import LLMEvaluator
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from models.transformer_model import TransformerModel
from models.icl_model import ICLModel
from util.loading import load_checkpoint, load_config

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--eval", type=str)
    parser.add_argument("--llm_eval", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--generate", type=bool, default=False)
    parser.add_argument("--device_id", type=int, default=-1)
    args = parser.parse_args()

    assert args.train or args.eval or args.llm_eval, "Must specify train, eval, or llm_eval"
    
    if args.llm_eval == "baseline":
        config = load_config("transformer")
        tokenizer = Tokenizer()
        config.vocab_size = tokenizer.vocab_size
        config.name = "baseline"
        model = TransformerModel(config) # Dummy model for compatibility
        splits = TinyStoriesDataset.get_splits(tokenizer, config.max_seq_len)
        llm_evaluator = LLMEvaluator(model, tokenizer, splits)
        llm_evaluator.run_baseline_eval()
        return
    
    if args.eval == "baseline":
        config = load_config("transformer")
        tokenizer = Tokenizer()
        config.vocab_size = tokenizer.vocab_size
        config.name = "baseline"
        model = TransformerModel(config) # Dummy model for compatibility
        splits = TinyStoriesDataset.get_splits(tokenizer, config.max_seq_len)
        evaluator = Evaluator(model, splits, tokenizer, checkpoint=None)
        evaluator.evaluate(do_generations=True)
        return
    
    if args.train:
        model_name = args.train
    elif args.eval:
        model_name = args.eval
    elif args.llm_eval:
        model_name = args.llm_eval
        
    config = load_config(model_name)
    
    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size
    
    if config.model_type == "transformer":
        model = TransformerModel(config)
    elif config.model_type == "icl":
        model = ICLModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    splits = TinyStoriesDataset.get_splits(tokenizer, config.max_seq_len)
    
    if args.train:
        checkpoint_type = args.checkpoint if args.checkpoint else "recent"
        checkpoint = load_checkpoint(model, checkpoint_type)
        trainer = Trainer(model, splits, tokenizer, checkpoint=checkpoint, device_id=args.device_id)
        trainer.train()
    else:
        checkpoint_type = args.checkpoint if args.checkpoint else "epoch_5" # For the paper, we used 5 epochs each 
        checkpoint = load_checkpoint(model, checkpoint_type)
        assert checkpoint is not None, f"Checkpoint not found: {checkpoint_type}"
        epoch = checkpoint["epoch"] if "epoch" in checkpoint else "N/A"
        print(f"Loaded checkpoint: {checkpoint_type} [{epoch} epochs]")
        
        if args.eval:
            evaluator = Evaluator(model, splits, tokenizer, checkpoint=checkpoint)
            evaluator.evaluate(do_generations=args.generate)
        elif args.llm_eval:
            llm_evaluator = LLMEvaluator(model, tokenizer, splits, checkpoint=checkpoint)
            llm_evaluator.run_llm_eval()
        

if __name__ == "__main__":
    main()