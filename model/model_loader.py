import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class PriceDirectionClassifier(nn.Module):
    """
    Single-task classifier for 3-class price direction prediction.

    Architecture:
    - Transformer (BERT/DistilBERT)
    - Dropout layer
    - Linear classification head (hidden_size -> 3 classes)
    """

    def __init__(self, model_name, dropout_rate):
        """
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased')
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
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
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token representation (first token)
        pooled = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

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

        # handle both transformer architectures, quack
        if hasattr(self.transformer, "encoder"):  # BERT
            layers = self.transformer.encoder.layer
        elif hasattr(self.transformer, "transformer"):  # DistilBERT
            layers = self.transformer.transformer.layer
        else:
            raise ValueError("Unknown transformer architecture")

        total_layers = len(layers)
        layers_to_unfreeze = list(range(total_layers - n_layers, total_layers))

        for layer_idx in layers_to_unfreeze:
            for param in layers[layer_idx].parameters():
                param.requires_grad = True

        print(
            f"Unfroze top {n_layers} layers (layers {layers_to_unfreeze}) out of {total_layers} total"
        )


class ModelLoader:
    """Helper class to load models and tokenizers with different configurations."""

    def __init__(self, config):
        """Load configuration from YAML file."""
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]

    def load_model_and_tokenizer(self):
        """
        Load model and tokenizer based on configuration.

        Returns:
            model: PriceDirectionClassifier or PEFT-wrapped model
            tokenizer: HuggingFace tokenizer
        """
        model_name = self.model_config["name"]
        print(f"Loading model: {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load base model
        dropout_rate = self.training_config["dropout_rate"]
        model = PriceDirectionClassifier(model_name, dropout_rate)

        # Apply LoRA or freezing strategy (mutually exclusive)
        use_lora = self.model_config.get("use_lora", False)

        if use_lora:
            print(f"Finetuing with Lora")
            model = self._apply_lora(model)
        else:
            freeze_strategy = self.model_config.get(
                "freeze_strategy", "partial"
            )  # set default argument to partial
            print(f"Freezing strategy: {freeze_strategy}")
            self._apply_freezing_strategy(model, freeze_strategy)

        print()
        self.print_trainable_parameters(model)

        return model, tokenizer

    def _apply_freezing_strategy(self, model, freezing_strategy):
        """
        Apply freezing strategy to model.

        Args:
            model: PriceDirectionClassifier instance
            freezing_strategy: 'frozen', 'partial', or 'full'
        """
        if freezing_strategy == "frozen":
            model.freeze_transformer()
        elif freezing_strategy == "partial":
            n_layers = self.model_config.get("num_frozen_layers", 2)
            model.unfreeze_top_layers(n_layers)
        elif freezing_strategy == "full":
            model.unfreeze_transformer()
        else:
            raise ValueError(
                f"Unknown freeze_strategy: {freezing_strategy}. Use 'frozen', 'partial', or 'full'"
            )

    def _apply_lora(self, model):
        """
        Apply LoRA to the transformer inside the model.

        Args:
            model: PriceDirectionClassifier instance

        Returns:
            model: PriceDirectionClassifier with LoRA-wrapped transformer
        """
        lora_config_dict = self.model_config.get("lora", {})

        # Determine correct target modules based on model architecture
        # BERT uses: attention.self.query, attention.self.value
        # DistilBERT uses: attention.q_lin, attention.v_lin
        model_name = self.model_config["name"]
        if "distilbert" in model_name.lower():
            default_targets = ["q_lin", "v_lin"]  # DistilBERT
        else:
            default_targets = ["query", "value"]  # BERT

        # Create LoRA configuration
        # Note: We don't specify task_type because we're wrapping the base transformer,
        # not the full classification model. The classifier head is separate.
        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 8),
            lora_alpha=lora_config_dict.get("lora_alpha", 16),
            target_modules=lora_config_dict.get("target_modules", default_targets),
            lora_dropout=lora_config_dict.get("lora_dropout", 0.1),
            bias=lora_config_dict.get("bias", "none"),
        )

        # Apply LoRA to the transformer, not the whole model
        model.transformer = get_peft_model(model.transformer, lora_config)

        return model

    def print_trainable_parameters(self, model):
        """
        Print trainable parameter statistics.

        Args:
            model: PyTorch model (regular or PEFT)
        """
        # check if it's a PEFT model, quack
        if hasattr(model, "print_trainable_parameters"):
            print("Parameter Statistics:")
            model.print_trainable_parameters()
        else:
            # count for non-PEFT models
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            trainable_pct = 100 * trainable_params / total_params

            print("Parameter Statistics:")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable %: {trainable_pct:.2f}%")

    def count_parameters(self, model):
        """
        Count total and trainable parameters.

        Args:
            model: PyTorch model

        Returns:
            (total_params, trainable_params): Tuple of integers
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
