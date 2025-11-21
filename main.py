# python main.py --sample --sample-size 100

import yaml
from pathlib import Path
import argparse
import torch
#from data.Gold.data_loader import DataLoader
from data.Stanford.data_loader_stanford import DataLoaderStanford
from model.model_loader import ModelLoader
from utils.training import Trainer


def load_config(config_path="configs/config.yaml"):
    """
    Load YAML configuration file

    This is the ONLY place we load config in the entire project.
    Config is then passed to all other modules.
    """
    # Always resolve path relative to this file's location (main.py)
    main_dir = Path(__file__).parent
    config_path = main_dir / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")
    return config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train and compare LLM models on classification task"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--sample", action="store_true", help="Use small sample dataset for testing"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to use in sample mode (default: 100)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # initialize data loader
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)
    data_loader = DataLoaderStanford(config)

    # Load or download data
    try:
        df = data_loader.load_csv()
        print("Loaded existing processed data")
    except FileNotFoundError:
        print("Downloading data from Kaggle...")
        df = data_loader.load_data()
        data_loader.save_csv(df)

    # Use sample if requested
    if args.sample:
        print(f"\nUsing sample of {args.sample_size} rows for testing")
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42).reset_index(
            drop=True
        )

    # split data
    train_df, val_df, test_df = data_loader.split_data(
        df, test_size=config["data"]["test_size"], val_size=config["data"]["val_size"]
    )

    # load model and tokenizer
    print("\n" + "=" * 60)
    print("Loading Model")
    print("=" * 60)
    model_loader = ModelLoader(config)

    # load model (config determines whether BERT or DistilBERT, LoRA or full fine-tuning)
    model, tokenizer = model_loader.load_model_and_tokenizer()

    # create dataloaders
    print("\n" + "=" * 60)
    print("Creating DataLoaders")
    print("=" * 60)
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        max_length=config["model"]["max_len"],
    )

    # initialize trainer
    print("\n" + "=" * 60)
    print("Initializing Trainer")
    print("=" * 60)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    # Train
    trainer.train()

    # Test
    trainer.test()

    # Finish W&B logging
    trainer.logger.finish()

    print("\n" + "=" * 60)
    print("Run Complete")


if __name__ == "__main__":
    main()
