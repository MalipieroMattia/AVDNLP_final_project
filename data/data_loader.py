import os
import pandas as pd
import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torch


class DataLoader:
    def __init__(self, config):
        self.config = config
        project_root = Path(__file__).parent.parent
        self.data_dir = (project_root / self.config['data'].get("data_dir", "data")).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = self.config['data']['test_size']
        self.val_size = self.config['data']["val_size"]
        self.max_length = self.config['model']['max_len']
        self.batch_size = self.config['training']['batch_size']

    def load_data(self):
        """Download dataset from Kaggle and load it into a DataFrame."""
        # Download dataset
        # Access config values as dictionary keys
        path = kagglehub.dataset_download(self.config['data']["kaggle_dataset"])

        # Find CSV file
        if self.config['data']["file_path"]:
            csv_path = os.path.join(path, self.config['data']["file_path"])
        else:
            # Auto-detect first CSV file
            csv_files = list(Path(path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            csv_path = str(csv_files[0])

        # Load data with pandas_kwargs from config
        df = pd.read_csv(csv_path, **self.config['data'].get("pandas_kwargs", {}))

        # Drop URL and Dates columns if they exist
        columns_to_drop = ["URL", "Dates", "Price Sentiment"]
        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )
        df = self.preprocess_data(df)
        return df

    def preprocess_data(self, df):
        """Preprocess the DataFrame (e.g., handle missing values)."""
        # Standardize column names: lowercase with underscores
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        return df
    

    def split_data(self, df, test_size, val_size, random_state=42):
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame with data
            test_size: Proportion for test set (0.2 = 20%)
            val_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            train_df, val_df, test_df
        """
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['price_direction']  # Stratify on single target column
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val['price_direction']
        )
        
        print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    

    def create_dataset(self, df, tokenizer, max_length):
        """
        Create a PyTorch Dataset from DataFrame.
        
        Args:
            df: DataFrame with 'text' column and 'price_direction' label
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            
        Returns:
            NewsDataset object
        """
        return NewsDataset(df, tokenizer, max_length)
    
    def create_dataloaders(self, train_df, val_df, test_df, tokenizer, batch_size, max_length):
        """
        Create PyTorch DataLoaders for train, val, and test sets.
        
        Args:
            train_df, val_df, test_df: DataFrames with 'text' and 'price_direction'
            tokenizer: HuggingFace tokenizer
            batch_size: Batch size for training
            max_length: Maximum sequence length
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset = self.create_dataset(train_df, tokenizer, max_length)
        val_dataset = self.create_dataset(val_df, tokenizer, max_length)
        test_dataset = self.create_dataset(test_df, tokenizer, max_length)
        
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
        return train_loader, val_loader, test_loader

    def save_csv(self, df, filename="processed_data.csv"):
        """Save DataFrame to data directory."""
        save_path = self.data_dir / filename
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
        return save_path

    def load_csv(self, filename="processed_data.csv"):
        """Load DataFrame from data directory."""
        load_path = self.data_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Not found: {load_path}")
        print(f"Loaded: {load_path}")
        return pd.read_csv(load_path)

    def load_sample_data(self, num_samples=5):
        """Load a small sample of the data for quick testing."""
        df = self.load_csv()
        return df.sample(n=num_samples).reset_index(drop=True)
    

class NewsDataset(Dataset):
    """PyTorch Dataset for news text with single 3-class price direction label."""
    
    def __init__(self, df, tokenizer, max_length):
        """
        Args:
            df: DataFrame with 'text' and 'price_direction' columns
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.texts = df['text'].tolist()
        self.labels = df['price_direction'].tolist()  # 0=up, 1=stable, 2=down
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dict with 'input_ids', 'attention_mask', and 'label'
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),      # Shape: [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Shape: [max_length]
            'label': torch.tensor(label, dtype=torch.long)      # Shape: scalar
        }
