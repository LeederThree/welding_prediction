import torch
import torch.nn as nn
from vit_pytorch import ViT


class Simple1DCNN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            )
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            )
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * (input_size // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG19(torch.nn.Module):
    def __init__(self, in_channel=1, classes=5):
        super(VGG19, self).__init__()
        self.feature = torch.nn.Sequential(

            torch.nn.Conv1d(in_channel, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.AdaptiveAvgPool1d(7)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(3584,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 3584)
        x = self.classifier(x)
        return x


class ViTModel(nn.Module):
    def __init__(self, image_size, num_classes):
        super(ViTModel, self).__init__()
        self.vit_model = ViT(
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            dim=128,
            depth=6,
            heads=12,
            mlp_dim=128,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        x_vit = self.vit_model(x)
        return x_vit


class FusionModel(nn.Module):
    def __init__(self, cnn_model, vit_model, num_classes):
        super(FusionModel, self).__init__()
        self.cnn_model = cnn_model
        self.vit_model = vit_model

        self.fc_fusion = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x_cnn, x_vit):
        x_cnn = self.cnn_model(x_cnn)
        x_vit = self.vit_model(x_vit)

        x_combined = torch.cat((x_cnn, x_vit), dim=1)
        x_combined = self.fc_fusion(x_combined)

        return x_combined
