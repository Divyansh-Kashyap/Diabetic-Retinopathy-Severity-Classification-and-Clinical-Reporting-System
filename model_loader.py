import torch
from retinal_model import load_retinal_model


def load_model(path="dr_model.pth"):
    """
    Load the Diabetic Retinopathy model using automatic EfficientNet variant detection.
    """
    model = load_retinal_model(path)
    model.eval()
    return model
