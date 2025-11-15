import os
import pandas as pd
import kagglehub
from pathlib import Path
from main import load_config


class DataLoader:
    def __init__(self):
        self.config = load_config("configs/config.yaml")
        self.config = load_config("configs/config.yaml")
        project_root = Path(__file__).parent.parent
        self.data_dir = (project_root / self.config.get("data_dir", "data")).resolve()
        self.data_dir = (project_root / self.config.get("data_dir", "data")).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Download dataset from Kaggle and load it into a DataFrame."""
        # Download dataset
        # Access config values as dictionary keys
        path = kagglehub.dataset_download(self.config["kaggle_dataset"])

        path = kagglehub.dataset_download(self.config["kaggle_dataset"])

        # Find CSV file
        if self.config["file_path"]:
            csv_path = os.path.join(path, self.config["file_path"])
        if self.config["file_path"]:
            csv_path = os.path.join(path, self.config["file_path"])
        else:
            # Auto-detect first CSV file
            csv_files = list(Path(path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            csv_path = str(csv_files[0])

        # Load data with pandas_kwargs from config
        df = pd.read_csv(csv_path, **self.config.get("pandas_kwargs", {}))
        df = pd.read_csv(csv_path, **self.config.get("pandas_kwargs", {}))

        # Drop URL and Dates columns if they exist
        columns_to_drop = ["URL", "Dates", "Price Sentiment"]
        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )
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

        df.columns = df.columns.str.lower().str.replace(" ", "_")

        # Example preprocessing: drop rows with missing values
        # df = df.dropna().reset_index(drop=True)
        # df = df.dropna().reset_index(drop=True)
        return df

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
