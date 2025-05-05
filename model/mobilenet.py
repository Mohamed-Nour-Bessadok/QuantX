import torch
from torchvision import models

def load_mobilenetv2(pretrained=True):
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
    model.eval()  # Set the model to evaluation mode
    return model
