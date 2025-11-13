"""
Model definitions for B-Free
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from peft import LoraConfig, get_peft_model, TaskType


class BFreeModel(nn.Module):
    """
    B-Free detector based on DINOv2 with registers
    Fine-tuned end-to-end for binary classification (real vs fake)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-giant-reg",
        num_classes: int = 2,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained DINOv2 model
        print(f"Loading pretrained model: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden dimension
        self.hidden_dim = self.backbone.config.hidden_size
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, pixel_values):
        """
        Forward pass
        
        Args:
            pixel_values: Tensor of shape (batch_size, 3, height, width)
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Get backbone features
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        
        # Use CLS token from last layer
        cls_token = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_dim)
        
        # Classify
        logits = self.classifier(cls_token)
        
        return logits
    
    def get_features(self, pixel_values):
        """Extract features without classification"""
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token


class BFreeModelLoRA(nn.Module):
    """
    B-Free detector with LoRA (Low-Rank Adaptation)
    Reduces trainable parameters by ~99% while maintaining performance
    
    LoRA Benefits:
    - Memory: ~10GB VRAM instead of 40GB for DINOv2-giant
    - Speed: 2-3x faster training
    - Storage: Checkpoints ~100MB instead of 4GB
    - Performance: Typically within 1-2% of full fine-tuning
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-with_registers-large",
        num_classes: int = 2,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained DINOv2 model
        print(f"Loading pretrained model with LoRA: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden dimension
        self.hidden_dim = self.backbone.config.hidden_size
        
        # Configure LoRA
        # Default: Apply LoRA to query and value projection matrices
        if target_modules is None:
            target_modules = ["query", "value"]  # Can also add "key", "dense"
        
        lora_config = LoraConfig(
            r=lora_r,  # Rank of low-rank matrices (higher = more capacity, default 16)
            lora_alpha=lora_alpha,  # Scaling factor (typically 2*r, default 32)
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",  # Don't adapt biases
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        # Apply LoRA to backbone
        self.backbone = get_peft_model(self.backbone, lora_config)
        
        # Print trainable parameters
        self.backbone.print_trainable_parameters()
        
        # Classification head (always trainable)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, pixel_values):
        """Forward pass with LoRA"""
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits
    
    def get_features(self, pixel_values):
        """Extract features without classification"""
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token
    
    def merge_and_unload(self):
        """
        Merge LoRA weights into base model for inference
        This eliminates LoRA overhead during deployment
        """
        self.backbone = self.backbone.merge_and_unload()
        return self


class BFreeModelLinearProbe(nn.Module):
    """
    B-Free detector with frozen backbone (linear probing only)
    Used for comparison in ablation studies
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-giant-reg",
        num_classes: int = 2
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained model
        print(f"Loading pretrained model: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get hidden dimension
        self.hidden_dim = self.backbone.config.hidden_size
        
        # Only train the classifier
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
        # Initialize
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, pixel_values):
        """Forward pass with frozen backbone"""
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
            cls_token = outputs.last_hidden_state[:, 0]
        
        logits = self.classifier(cls_token)
        return logits


def build_model(config: dict, linear_probe: bool = False, use_lora: bool = False):
    """
    Build B-Free model based on configuration
    
    Args:
        config: Configuration dictionary
        linear_probe: If True, use linear probing only
        use_lora: If True, use LoRA for parameter-efficient fine-tuning
    
    Returns:
        model: BFree model
    """
    model_config = config['training']
    
    if use_lora and linear_probe:
        raise ValueError("Cannot use both LoRA and linear probe simultaneously")
    
    if use_lora:
        # Get LoRA config if specified
        lora_config = config['training'].get('lora', {})
        model = BFreeModelLoRA(
            model_name=model_config['model_name'],
            num_classes=model_config['num_classes'],
            lora_r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', None)
        )
    elif linear_probe:
        model = BFreeModelLinearProbe(
            model_name=model_config['model_name'],
            num_classes=model_config['num_classes']
        )
    else:
        model = BFreeModel(
            model_name=model_config['model_name'],
            num_classes=model_config['num_classes'],
            freeze_backbone=False
        )
    
    return model
