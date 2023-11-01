import torchvision.models as models
import torch.nn as nn


resnet18 = models.resnet18(pretrained=True)
num_classes = 10
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
