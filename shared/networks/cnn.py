import torch.nn as nn

class LightweightCNN(nn.Module):
    """Lightweight CNN classifier optimized for small datasets"""

    def __init__(self, num_classes):
        super(LightweightCNN, self).__init__()

        # Much simpler architecture
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            # Second block
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            # Third block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Global Average Pooling instead of large FC layer
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Much smaller classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
