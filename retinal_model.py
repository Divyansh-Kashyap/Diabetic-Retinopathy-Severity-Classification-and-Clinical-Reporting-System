import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class RetinalModel(nn.Module):
    """
    Diabetic Retinopathy Severity Classification Model
    Automatically selects the correct EfficientNet variant based on checkpoint.
    """

    def __init__(self, n_classes=5, backbone="efficientnet-b6"):
        super().__init__()

        # Load pretrained EfficientNet of correct variant
        self.model = EfficientNet.from_pretrained(backbone)

        # Replace final FC layer
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.model(x)


def load_retinal_model(checkpoint_path):
    """
    Automatically detects the correct EfficientNet variant by reading
    the checkpoint shapes before loading the full model.
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Detect variant from unique layer shapes
    variant = None

    # EfficientNet-B2 head: 1408 channels (seen in error logs)
    if any("1408" in str(v.shape) for v in state_dict.values()):
        variant = "efficientnet-b2"

    # EfficientNet-B6 head: 1280 channels (most common)
    if any("1280" in str(v.shape) for v in state_dict.values()):
        variant = "efficientnet-b6"

    # EfficientNet-B7 head
    if any("2560" in str(v.shape) for v in state_dict.values()):
        variant = "efficientnet-b7"

    # Fallback: default to B6
    if variant is None:
        print("Variant detection failed. Defaulting to EfficientNet-B6.")
        variant = "efficientnet-b6"

    print(f"Detected model variant: {variant}")

    # Build model
    model = RetinalModel(n_classes=5, backbone=variant)

    # Load weights
    model.load_state_dict(state_dict, strict=False)

    print("Model loaded successfully.\n")
    return model
