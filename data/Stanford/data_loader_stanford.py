# ...existing code...
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torch

class DataLoaderStanford:
    """
    Data loader for locally stored Stanford-style dataset (txt/csv).
    Expects config dict with keys similar to Gold dataloader:
      config['data']['file_path']  -> relative or absolute path to the .txt/.csv file
      config['data'].get('sep')   -> optional separator (default: '\t')
      config['data'].get('pandas_kwargs', {}) -> passed to pd.read_csv
      config['data']['test_size'], config['data']['val_size']
      config['model']['max_len'], config['training']['batch_size']
      config['data'].get('data_dir') -> optional output data dir
    The file is read with pandas; if parsing fails the file is read line-by-line
    and each line becomes one 'text' entry.
    """
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
        """
        Load the local file specified in config. Defaults to TSV (sep='\t').
        Builds 'text' from sentence1/sentence2 and renames 'gold_label' -> 'price_direction'.
        """
        file_path = Path(self.config['data']['file_path'])
        if not file_path.is_absolute():
            file_path = (Path(__file__).parent.parent / file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        pandas_kwargs = self.config['data'].get('pandas_kwargs', {})
        sep = self.config['data'].get('sep', '\t')  # use tab by default for your .txt

        # Read with pandas (TSV/CSV); fallback to line-per-example handled by preprocess if needed
        df = pd.read_csv(file_path, sep=sep, engine='python', encoding=self.config['data'].get('encoding', 'utf-8'), **pandas_kwargs)

        # Build single 'text' column: prefer concatenation of sentence1 and sentence2
        if 'sentence1' in df.columns and 'sentence2' in df.columns:
            df['text'] = df['sentence1'].astype(str).str.strip() + ' [SEP] ' + df['sentence2'].astype(str).str.strip()
        elif 'sentence1' in df.columns:
            df['text'] = df['sentence1'].astype(str)
        elif 'sentence2' in df.columns:
            df['text'] = df['sentence2'].astype(str)

        # Optional: map string labels to integers via config['data']['label_map']
        # e.g. config['data']['label_map'] = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


        df = self.preprocess_data(df)
        return df

    def preprocess_data(self, df):
        # drop columns that are not needed for
        df = df.drop(columns=['sentence1','sentence2','sentence1_binary_parse', 'sentence2_binary_parse','sentence1_parse','sentence2_parse','pairID','captionID','label1','label2','label3','label4','label5'])

        #rename gold_label to label
        if 'gold_label' in df.columns:
            df = df.rename(columns={'gold_label': 'label'})

        # Remove rows with invalid labels
        df = df[df['label'] != '-']
        
        label_map = self.config['data'].get('label_map')
        df['label'] = df['label'].map(label_map)
        #df['label'] = df['label'].fillna(-1).astype(int)

        # Reset index
        return df.reset_index(drop=True)

    def split_data(self, df, test_size=None, val_size=None, random_state=42):
        test_size = test_size if test_size is not None else self.test_size
        val_size = val_size if val_size is not None else self.val_size

        # If labels are placeholders (-1) we cannot stratify; fall back to non-stratified split
        stratify_col = None
        if 'label' in df.columns and df['label'].nunique() > 1 and (df['label'] != -1).all():
            stratify_col = df['label']

        # First split
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        # Second split (train vs val) - adjust val size relative to remaining
        val_size_adjusted = val_size / (1 - test_size)
        stratify_train = train_val['label'] if stratify_col is not None else None

        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_train
        )

        print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def create_dataset(self, df, tokenizer, max_length):
        return NewsDatasetStanford(df, tokenizer, max_length)

    def create_dataloaders(self, train_df, val_df, test_df, tokenizer, batch_size=None, max_length=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        max_length = max_length if max_length is not None else self.max_length

        train_dataset = self.create_dataset(train_df, tokenizer, max_length)
        val_dataset = self.create_dataset(val_df, tokenizer, max_length)
        test_dataset = self.create_dataset(test_df, tokenizer, max_length)

        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader

    def save_csv(self, df, filename="stanford_processed.csv"):
        save_path = self.data_dir / filename
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
        return save_path

    def load_csv(self, filename="stanford_processed.csv"):
        load_path = self.data_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Not found: {load_path}")
        print(f"Loaded: {load_path}")
        return pd.read_csv(load_path)

    def load_sample_data(self, num_samples=5):
        df = self.load_csv()
        return df.sample(n=num_samples).reset_index(drop=True)


class NewsDatasetStanford(Dataset):
    """
    PyTorch Dataset for text + single integer label 'price_direction'.
    If labels are -1 they will still be returned (model/train loop must handle).
    """
    def __init__(self, df, tokenizer, max_length):
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
# ...existing code...