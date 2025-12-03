import torch
import cv2
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((380, 380)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(model, image):
    model.eval()
    img_tensor = transform(image).unsqueeze(0)

    target_layer = model.model._blocks[-1]._project_conv

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = torch.argmax(output)

    model.zero_grad()
    output[0, class_idx].backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grad = gradients[0].mean(dim=[2, 3], keepdim=True)
    activation = activations[0]

    heatmap = torch.relu((activation * grad).sum(dim=1)).squeeze().detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    return overlay
