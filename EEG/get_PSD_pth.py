# load packages
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# load data
with open('/root/autodl-tmp/mdd/data/EEG/Emo-features/PSDX_data.pkl', 'rb') as f:
    PSD_x = pickle.load(f)

with open('/root/autodl-tmp/mdd/data/EEG/Emo-features/PSDy_data.pkl', 'rb') as f:
    PSD_y = pickle.load(f)

# transpose
PSD_x = PSD_x.transpose(0, 3, 1, 2)



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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def train_model(PSD_x, PSD_y, label_indices, dataset_name):
    for i in label_indices:
        print(f'Training for label {i}')
        
        in_channels = 5
        num_blocks = 2
        block_channels = [64, 128, 256, 512]
        
        model = ResNetRegressor(in_channels, num_blocks, block_channels).to(device)
        
        PSD_x_tensor = torch.Tensor(PSD_x).to(device)
        PSD_y_tensor = torch.Tensor(PSD_y[:, i]).to(device) 

        dataset = TensorDataset(PSD_x_tensor, PSD_y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 1000
        for epoch in range(num_epochs):
            running_loss = 0.0
            for data in dataloader:
                inputs, labels = data

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print('Epoch %d, Loss: %.4f' % (epoch + 1, running_loss / len(dataloader)))

        torch.save(model.state_dict(), f'{dataset_name}_{i}.pth')

label_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
train_model(PSD_x, PSD_y, label_indices, 'PSD')

print('Finished Training')