import torch.nn as nn
class CNN1DModel(nn.Module):
    def __init__(self, conv_archs, num_classes, batch_size, input_channels=1):
        super().__init__()
        self.batch_size = batch_size
        self.conv_arch = conv_archs
        self.input_channels = input_channels
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool1d(9)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, num_classes),
        )

    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(self.batch_size, 1, 1024)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(self.batch_size, -1)
        x = self.classifier(x)
        return x
