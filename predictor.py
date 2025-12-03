import torch
import cv2
import numpy as np
import torchvision.transforms as T

SEVERITY = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((380, 380)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_stage(model, image):
    model.eval()
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        score, idx = torch.max(probs, 1)

    return SEVERITY[idx.item()], score.item()
