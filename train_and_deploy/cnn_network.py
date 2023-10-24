
# Neural Network architecture definitions

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

class simpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv64 = nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv128 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2))
        self.conv256 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))

        #SPATIAL DIMENSION FORMULA (Assume no padding)
        #(Input height - kernel height) / (stride + 1)

        #200x200
        #((200 - 5) / 2) + 1 = 98
        #((98 - 5) / 2) + 1 = 47
        #((47 - 3)) / 1) + 1 = 45

        self.fc1 = nn.Linear(256*45*45, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        

    def forward(self, x):               
        x = self.relu(self.conv64(x))  
        x = self.relu(self.conv128(x))  
        x = self.relu(self.conv256(x))  

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        return x

# Dokney docs --> parts --> keras and fastai (uses PyTorch)
# donkeycar (github) --> donkeycar --> parts --> keras.py and fastai.py (try fastai first)
# https://github.com/autorope/donkeycar/blob/main/donkeycar/parts/fastai.py
class DonkeyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.fc1 = nn.Linear(64*30*30, 128)  # (64*30*30, 128) for 300x300 images
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):               #   300x300                     #  120x160 images
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148     |     (120-5)/2+1 = 58   (160-5)/2+1 = 78
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72      |     (58 -5)/2+1 = 27   (78 -5)/2+1 = 37
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34     |     (27 -5)/2+1 = 12   (37 -5)/2+1 = 17
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32         |     12 - 3 + 1  = 10   17 - 3 + 1  = 15
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30         |     10 - 3 + 1  = 8    15 - 3 + 1  = 13

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseNetRC(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super(DenseNetRC, self).__init__()

        # Define the initial convolution layer for 200x200 images
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Define dense blocks and transition layers (adjust as needed)
        # Add more dense blocks for greater depth
        self.dense_block1 = self._make_dense_block(3, 6, growth_rate=16)
        self.transition1 = self._make_transition_layer(256)

        # Final batch normalization
        self.features.add_module('norm5', nn.BatchNorm2d(256))

        # Classifier for self-driving control (adjust output size)
        self.classifier = nn.Linear(256, num_classes)

        # Initialize weights and biases
        self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _make_dense_block(self, num_input_features, num_layers, growth_rate):
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_input_features):
        return nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_input_features // 2, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))


