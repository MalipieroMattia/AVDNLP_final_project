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

        # Drop URL and Dates columns if they exist
        columns_to_drop = ['URL', 'Dates']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        df = self.preprocess_data(df)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the DataFrame (e.g., handle missing values)."""
        # Standardize column names: lowercase with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Example preprocessing: drop rows with missing values
        df = df.dropna().reset_index(drop=True)
        return df   