from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from main import load_config


class MultiTaskClassifier(nn.Module):
    """Multi-task classifier for 3 binary classification tasks."""
    
    def __init__(self, config):
        super().__init__()
        model_name = config['model']['distilbert']
        
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = config['model']['hidden_size']
        dropout = config['model']['dropout_rate']
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Three binary classification heads
        self.direction_head = nn.Linear(hidden_size, 2)
        self.comparison_head = nn.Linear(hidden_size, 2)
        self.future_info_head = nn.Linear(hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token (first token)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        
        # Get predictions from each head
        return {
            'direction': self.direction_head(pooled),
            'comparison': self.comparison_head(pooled),
            'future_info': self.future_info_head(pooled)
        }
    
    def freeze_transformer(self):
        """Freeze all transformer layers (train only classification heads)."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Transformer frozen - only training classification heads")
    
    def unfreeze_transformer(self):
        """Unfreeze all transformer layers (full fine-tuning)."""
        for param in self.transformer.parameters():
            param.requires_grad = True
        print("Transformer unfrozen - full fine-tuning enabled")
    
    def unfreeze_top_layers(self, n_layers=2):
        """
        Unfreeze only the top N layers of the transformer.
        
        Args:
            n_layers: Number of top layers to unfreeze
        """
        # First freeze everything
        self.freeze_transformer()
        
        # DistilBERT has 6 transformer layers
        # Unfreeze the last n_layers
        total_layers = len(self.transformer.transformer.layer)
        layers_to_unfreeze = list(range(total_layers - n_layers, total_layers))
        
        for layer_idx in layers_to_unfreeze:
            for param in self.transformer.transformer.layer[layer_idx].parameters():
                param.requires_grad = True
        
        print(f"Unfroze top {n_layers} layers (out of {total_layers})")

class ModelLoader:
    def __init__(self):
        self.config = load_config('configs/config.yaml')

    def load_model_and_tokenizer(self, use_lora=False, freeze_strategy='frozen'):
        """Load model and tokenizer based on config."""
        model_name = self.config['model']['distilbert']
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load multi-task model
        model = MultiTaskClassifier(self.config)

        # Apply freezing strategy
        if freeze_strategy == 'frozen':
            model.freeze_transformer()
        elif freeze_strategy.startswith('partial-'):
            n_layers = int(freeze_strategy.split('-')[1])
            model.unfreeze_top_layers(n_layers)
        elif freeze_strategy == 'full':
            model.unfreeze_transformer()
        else:
            raise ValueError(f"Unknown freeze_strategy: {freeze_strategy}") 

        # Apply LoRA if specified
        if use_lora:
            self._apply_lora(model)

        return model, tokenizer
    
    def _apply_lora(self, model):
        """Apply LoRA to the model if specified in config."""
        # TODO: Implement in next step
        raise NotImplementedError("LoRA not yet implemented")
        return model

    def count_parameters(self, model):
        """Count total and trainable parameters in the model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    


    