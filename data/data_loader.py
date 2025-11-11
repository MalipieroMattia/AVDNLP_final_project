import os
import pandas as pd
import kagglehub
from pathlib import Path

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        """Download dataset from Kaggle and load it into a DataFrame."""
        # Download dataset
        path = kagglehub.dataset_download(self.config.kaggle_dataset)
        
        # Find CSV file
        if self.config.file_path:
            csv_path = os.path.join(path, self.config.file_path)
        else:
            # Auto-detect first CSV file
            csv_files = list(Path(path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            csv_path = str(csv_files[0])
        
        # Load data
        df = pd.read_csv(csv_path, **self.config.pandas_kwargs)
        return df