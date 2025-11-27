import torch
import torch.nn as nn

class MPID(nn.Module):
    def __init__(self, dropout=0.5, num_classes=5, eps=1e-05, running_stats=False):
        super(MPID, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(64, 64),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.GroupNorm(64, 64),
            nn.AvgPool2d(2, padding=1),

            nn.Conv2d(64, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(96, 96),
            nn.Conv2d(96, 96, 3, stride=1),
            nn.ReLU(),
            nn.GroupNorm(96, 96),
            nn.AvgPool2d(2, padding=1),

            nn.Conv2d(96, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(128, 128),
            nn.Conv2d(128, 128, 3, stride=1),
            nn.ReLU(),
            nn.GroupNorm(128, 128),
            nn.AvgPool2d(2, padding=1),

            nn.Conv2d(128, 160, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(160, 160),
            nn.Conv2d(160, 160, 3, stride=1),
            nn.ReLU(),
            nn.GroupNorm(160, 160),
            nn.AvgPool2d(2, padding=1),

            nn.Conv2d(160, 192, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(192, 192),
            nn.Conv2d(192, 192, 3, stride=1),
            nn.GroupNorm(192, 192),
            nn.AvgPool2d(2, padding=1)
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(192 * 8 * 8, 192 * 8),
            nn.Dropout(dropout),
            nn.Linear(192 * 8, 192),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

