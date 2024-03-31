import torch.nn as nn
import torch
from torchsummary import summary

class TumorCNN(nn.Module):
    def __init__(self, in_channels=1, num_filters=8, classes=2):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(num_filters*2*32*32, classes)
        )

    def forward(self, x):
        return self.cnn_block(x)

if __name__ == "__main__":
    model = TumorCNN()
    x = torch.randn(64, 1, 128, 128)
    print(model(x).shape)
    summary(model, (1, 128, 128))
