import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3).cuda()
        self.conv2 = nn.Conv2d(32, 64, 3).cuda()
        self.conv3 = nn.Conv2d(64, 128, 3).cuda()
        self.dropout = nn.Dropout(0.25).cuda()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 240, 224).cuda()
            x = F.relu(F.max_pool2d(self.conv1(dummy_input), 3))
            x = F.relu(F.max_pool2d(self.conv2(x), 3))
            x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
            self.conv_output_size = x.numel()

        self.fc1 = nn.Linear(self.conv_output_size, 256).cuda()
        self.fc2 = nn.Linear(256, 128).cuda()
        self.fc3 = nn.Linear(128, 9).cuda()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2(x), 3))
        x = self.dropout(F.relu(F.max_pool2d(self.conv3(x), 3, 2)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
