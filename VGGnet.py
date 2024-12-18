import torch
from torch import nn

CONFIGS = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}

class VGG(nn.Module):
    def __init__(self, config, batch_norm=False, num_classes=1000, init_weights=True, dropout=0.5):
        super(VGG, self).__init__()
        self.features = self.make_layers(config, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self.init_weight()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self, config, batch_norm):
        layers = []
        in_channels = 3
        for layer in config:
            if layer == "M":  # MaxPooling layer
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:  # Convolutional layer
                conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = layer
        return nn.Sequential(*layers)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

model = VGG(CONFIGS["D"], batch_norm=True, num_classes=1000)
from torchinfo import summary
summary(model, input_size=(2, 3, 224, 224))
