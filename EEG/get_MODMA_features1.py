import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


psd_tensor = np.load('psd_tensor.npy')
de_tensor = np.load('de_tensor.npy')
psd_tensor = np.transpose(psd_tensor, (0, 2, 3, 1))
de_tensor = np.transpose(de_tensor, (3, 2, 0, 1))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNetRegressor(nn.Module):
    def __init__(self, in_channels, num_blocks, block_channels):
        super(ResNetRegressor, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, block_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(block_channels[0])
        self.relu = nn.LeakyReLU(inplace=True)

        self.layers = self._make_layers(block_channels, num_blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block_channels[-1], 1)

    def _make_layers(self, block_channels, num_blocks):
        layers = []
        in_channels = block_channels[0]

        for i, out_channels in enumerate(block_channels):
            stride = 1 if i == 0 else 2
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


in_channels = 5
num_blocks = 2
block_channels = [64, 128, 256, 512]


model = ResNetRegressor(in_channels, num_blocks, block_channels)

model_dir = '/Users/chenxin/Desktop/MDD/data/EEG/EEG-features/DE/'

results = pd.DataFrame()


for file_name in os.listdir(model_dir):
    if file_name.endswith('.pth'):
        file_path = os.path.join(model_dir, file_name)
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            inputs = torch.tensor(de_tensor, dtype=torch.float32)
            outputs = model(inputs)

        results[file_name] = outputs.squeeze().numpy()

results.to_excel('de_model_outputs.xlsx', index=False)



model_dir = '/Users/chenxin/Desktop/MDD/data/EEG/EEG-features/PSD/'
results = pd.DataFrame()

for file_name in os.listdir(model_dir):
    if file_name.endswith('.pth'):
        file_path = os.path.join(model_dir, file_name)
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            inputs = torch.tensor(psd_tensor, dtype=torch.float32)
            outputs = model(inputs)

        results[file_name] = outputs.squeeze().numpy()

results.to_excel('psd_model_outputs.xlsx', index=False)