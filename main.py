import yaml
from pathlib import Path
import argparse

def load_config(config_path='configs/config.yaml'):
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
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {config_path}")
    return config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train and compare LLM models on classification task'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file (default: configs/config.yaml)'
    )
    args = parser.parse_args()
    config = load_config(args.config)

if __name__ == "__main__":
    main()