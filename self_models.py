import torch
import torch.nn as nn
from torchvision import models

# Ver_1_conv_4_fc_3
# class fake_vs_real_Model(nn.Module):
#     def __init__(self):
#         super(fake_vs_real_Model, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)  # out_size = （300 - 5 + 0）/ 1 + 1  = 296
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 2)  # 296 / 2 = 148
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=0)  # out_size = （148 - 5 + 0）/ 1 + 1  = 144
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2, 2)  # 144 / 2 = 72
#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=26, kernel_size=5, padding=0)  # out_size = （72 - 5 + 0）/ 1 + 1  = 68
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(2, 2)  # 68 / 2 = 34
#         self.conv4 = nn.Conv2d(in_channels=26, out_channels=32, kernel_size=3, padding=1)  # out_size = （34 - 3 + 2）/ 1 + 1  = 34
#         self.relu4 = nn.ReLU()
#         self.pool4 = nn.MaxPool2d(2, 2)  # 34 / 2 = 17
#         self.fc1 = nn.Linear(32 * 17 * 17, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 2)
#         self.output = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = self.pool4(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.output(x)
#
#         return x


# Ver_2_conv_4_fc_3
# class fake_vs_real_Model(nn.Module):
#     def __init__(self):
#         super(fake_vs_real_Model, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)  # out_size = （300 - 5 + 0）/ 1 + 1  = 296
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 2)  # 296 / 2 = 148
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=0)  # out_size = （148 - 5 + 0）/ 1 + 1  = 144
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2, 2)  # 144 / 2 = 72
#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=26, kernel_size=5, padding=0)  # out_size = （72 - 5 + 0）/ 1 + 1  = 68
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(2, 2)  # 68 / 2 = 34
#         self.conv4 = nn.Conv2d(in_channels=26, out_channels=32, kernel_size=3, padding=1)  # out_size = （34 - 3 + 2）/ 1 + 1  = 34
#         self.relu4 = nn.ReLU()
#         self.pool4 = nn.MaxPool2d(2, 2)  # 34 / 2 = 17
#         self.fc1 = nn.Linear(32 * 17 * 17, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 2)
#         self.output = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.pool3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.pool4(x)
#         x = self.relu4(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.output(x)
#
#         return x


# Ver_3_conv_4_fc_5
# class fake_vs_real_Model(nn.Module):
#     def __init__(self):
#         super(fake_vs_real_Model, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)  # out_size = （300 - 5 + 0）/ 1 + 1  = 296
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 2)  # 296 / 2 = 148
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=0)  # out_size = （148 - 5 + 0）/ 1 + 1  = 144
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2, 2)  # 144 / 2 = 72
#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=26, kernel_size=3, padding=1)  # out_size = （72 - 3 + 2）/ 1 + 1  = 72
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(2, 2)  # 72 / 2 = 36
#         self.conv4 = nn.Conv2d(in_channels=26, out_channels=32, kernel_size=3, padding=1)  # out_size = （36 - 3 + 2）/ 1 + 1  = 36
#         self.relu4 = nn.ReLU()
#         self.pool4 = nn.MaxPool2d(2, 2)  # 36 / 2 = 18
#         self.fc1 = nn.Linear(32 * 18 * 18, 512)
#         self.fc2 = nn.Linear(512, 640)
#         self.fc3 = nn.Linear(640, 384)
#         self.fc4 = nn.Linear(384, 128)
#         self.fc5 = nn.Linear(128, 2)
#         self.output = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.pool3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.pool4(x)
#         x = self.relu4(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.fc4(x)
#         x = self.fc5(x)
#         x = self.output(x)
#
#         return x


# Ver_4_conv_2_fc_3
# class fake_vs_real_Model(nn.Module):
#     def __init__(self):
#         super(fake_vs_real_Model, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)  # out_size = （300 - 5 + 0）/ 1 + 1  = 296
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 2)  # 296 / 2 = 148
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3, padding=1)  # out_size = （148 - 3 + 2）/ 1 + 1  = 148
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2, 2)  # 148 / 2 = 74
#         self.fc1 = nn.Linear(24 * 74 * 74, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 2)
#         self.output = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.relu2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.output(x)
#
#         return x


# Ver_7_conv_4_fc_6
class fake_vs_real_Model(nn.Module):
    def __init__(self):
        super(fake_vs_real_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)  # out_size = （300 - 5 + 0）/ 1 + 1  = 296
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 296 / 2 = 148
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=0)  # out_size = （148 - 5 + 0）/ 1 + 1  = 144
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 144 / 2 = 72
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=26, kernel_size=5, padding=0)  # out_size = （72 - 5 + 0）/ 1 + 1  = 68
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 68 / 2 = 34
        self.conv4 = nn.Conv2d(in_channels=26, out_channels=32, kernel_size=3, padding=1)  # out_size = （34 - 3 + 2）/ 1 + 1  = 34
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)  # 34 / 2 = 17
        self.fc1 = nn.Linear(32 * 17 * 17, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.output(x)

        return x


