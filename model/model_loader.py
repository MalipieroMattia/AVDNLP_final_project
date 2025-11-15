import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



class PriceDirectionClassifier(nn.Module):
    """
    Single-task classifier for 3-class price direction prediction.
    
    Architecture:
    - DistilBERT transformer (pre-trained)
    - Dropout layer
    - Linear classification head (768 -> 3 classes)
    """
    
    def __init__(self,config):
        """
        Args:
            config: Configuration dict with model settings
        """
        self.config = config
        super().__init__()
        model_name = self.config['model']['distilbert']
        
        # Load pre-trained DistilBERT
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from transformer config (usually 768 for DistilBERT)
        hidden_size = self.transformer.config.hidden_size
        dropout_rate = self.config['training']['dropout_rate']
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head: 3 classes (up=0, stable=1, down=2)
        self.classifier = nn.Linear(hidden_size, 3)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs, shape [batch_size, seq_length]
            attention_mask: Attention mask, shape [batch_size, seq_length]
            
        Returns:
            logits: Raw scores for 3 classes, shape [batch_size, 3]
        """
        # Pass through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token representation (first token)
        # This represents the entire sequence
        pooled = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, 768]
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.classifier(pooled)  # Shape: [batch_size, 3]
        
        return logits
    
    def freeze_transformer(self):
        """Freeze all transformer parameters (only train classifier head)."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Froze all transformer layers")
    
    def unfreeze_transformer(self):
        """Unfreeze all transformer parameters (full fine-tuning)."""
        for param in self.transformer.parameters():
            param.requires_grad = True
        print("Unfroze all transformer layers")
    
    def unfreeze_top_layers(self, n_layers):
        """
        Unfreeze only the top N transformer layers (partial fine-tuning).
        
        Args:
            n_layers: Number of top layers to unfreeze
        """
        # First freeze everything
        self.freeze_transformer()
        
        # DistilBERT has 6 layers (0-5)
        total_layers = len(self.transformer.transformer.layer)
        
        # Unfreeze top n_layers (e.g., layers 4 and 5 if n_layers=2)
        layers_to_unfreeze = list(range(total_layers - n_layers, total_layers))
        
        for layer_idx in layers_to_unfreeze:
            for param in self.transformer.transformer.layer[layer_idx].parameters():
                param.requires_grad = True
        
        print(f"Unfroze top {n_layers} layers (layers {layers_to_unfreeze}) out of {total_layers} total")


class ModelLoader:
    """Helper class to load models and tokenizers with different configurations."""
    
    def __init__(self,config):
        """Load configuration from YAML file."""
        self.config = config
        self.freeze_strategy = self.config['model'].get('freeze_strategy', 'frozen')
        self.frozen_layers = self.config['model']['num_frozen_layers']
        self.use_lora = self.config['model']['use_lora']

    def load_model_and_tokenizer(self, use_lora=False, freeze_strategy=None):
        """
        Load model and tokenizer based on configuration.
        
        Args:
            use_lora: Whether to apply LoRA (not yet implemented)
            freeze_strategy: One of:
                - 'frozen': Freeze all transformer layers
                - 'partial': Unfreeze top N layers (N from config)
                - 'full': Unfreeze all layers
                
        Returns:
            model: PriceDirectionClassifier instance
            tokenizer: HuggingFace tokenizer
        """
        model_name = self.config['model']['distilbert']
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = PriceDirectionClassifier(self.config)

        # Apply freezing strategy
        if freeze_strategy == 'frozen':
            model.freeze_transformer()
        elif freeze_strategy == 'partial':
            # Get number of layers from config
            n_layers = self.frozen_layers
            model.unfreeze_top_layers(n_layers)
        elif freeze_strategy == 'full':
            model.unfreeze_transformer()
        else:
            raise ValueError(f"Unknown freeze_strategy: {freeze_strategy}. Use 'frozen', 'partial', or 'full'")

        # Apply LoRA if specified (future work)
        if use_lora:
            self._apply_lora(model)

        return model, tokenizer
    
    def _apply_lora(self, model):
        """Apply LoRA to the model (to be implemented)."""
        raise NotImplementedError("LoRA not yet implemented")
    
    def count_parameters(self, model):
        """
        Count total and trainable parameters in the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            (total_params, trainable_params): Tuple of integers
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params