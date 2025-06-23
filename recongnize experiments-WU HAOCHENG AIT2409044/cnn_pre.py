import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import torch.nn.functional as F
import math
import random
from PIL import Image
import numpy as np

class FlexibleCNN(nn.Module):
    def __init__(self, num_classes=37, layers_enabled=None, kernel_sizes=None):
        super(FlexibleCNN, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 5, 3, 3, 3, 3, 3]
        if layers_enabled is None:
            layers_enabled = [1, 1, 1, 1, 1, 1, 1, 1]
        self.layers_enabled = layers_enabled
        self.kernel_sizes = kernel_sizes
        self.channel_plan = [16, 32, 64, 128, 128, 256, 256, 512]
        self.convs = nn.ModuleList()
        in_channels = 1
        for i in range(len(self.channel_plan)):
            if self.layers_enabled[i]:
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channel_plan[i],
                    kernel_size=self.kernel_sizes[i],
                    padding=self.kernel_sizes[i]//2
                )
                self.convs.append(conv)
                in_channels = self.channel_plan[i]
            else:
                self.convs.append(None)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x
        for i, conv in enumerate(self.convs):
            if conv is not None:
                out = F.relu(conv(out))
                if i % 2 == 1:
                    out = self.pool(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 打开图片
    input_tensor = transform(image).unsqueeze(0)   # 增加 batch 维度
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class


model = torch.load(r'C:\Users\吴\Desktop\code\python\principle\final_cnn_best_100.pth', map_location=torch.device('cpu'))
model.eval()
# --------- 图片预处理 ----------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img_path=r"C:\Users\吴\Desktop\code\python\principle\bmp\k\img021-00010.png"
classes=["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f",
         "g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
predicted = predict_image(img_path)
print(f"Predicted class: {classes[predicted]}")
