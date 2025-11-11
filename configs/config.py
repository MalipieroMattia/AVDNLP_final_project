from dataclasses import dataclass, field
import random
import os
from typing import Optional, Dict, List
import numpy as np

@dataclass
class Config:
    seed: int = 42
    kaggle_dataset: str = "ankurzing/sentiment-analysis-in-commodity-market-gold"
    file_path: str = ""  # relative path inside the dataset or empty for default
    pandas_kwargs: Optional[Dict] = field(default_factory=dict)
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    
    # Weights & Biases configuration
    project: str = ""
    tags: List[str] = None
    api_key: str = ""
    entity: str = "jojs-it-universitetet-i-k-benhavn"

    def apply_seed(self) -> None:
        """Apply seed to Python, NumPy and (if installed) PyTorch."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except Exception:
            pass
