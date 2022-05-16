import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import BACKBONES


@BACKBONES.register_module()
class Unet(nn.Module):

    def __init__(self, nf=64):
        super(Unet, self).__init__()

        self.conv1_1 = nn.Conv2d(4, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)

        self.upv5 = nn.ConvTranspose2d(nf * 8, nf * 4, 2, stride=2)
        self.conv5_1 = nn.Conv2d(nf * 8, nf * 4, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(nf * 4, nf * 2, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf * 4, nf * 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(nf * 2, nf, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(nf, 4, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = F.leaky_relu(self.conv1_1(x), 0.2)
        conv1 = F.leaky_relu(self.conv1_2(conv1), 0.2)
        pool1 = self.pool1(conv1)

        conv2 = F.leaky_relu(self.conv2_1(pool1), 0.2)
        conv2 = F.leaky_relu(self.conv2_2(conv2), 0.2)
        pool2 = self.pool2(conv2)

        conv3 = F.leaky_relu(self.conv3_1(pool2), 0.2)
        conv3 = F.leaky_relu(self.conv3_2(conv3), 0.2)
        pool3 = self.pool3(conv3)

        conv4 = F.leaky_relu(self.conv4_1(pool3), 0.2)
        conv4 = F.leaky_relu(self.conv4_2(conv4), 0.2)

        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv3], 1)
        conv5 = F.leaky_relu(self.conv5_1(up5), 0.2)
        conv5 = F.leaky_relu(self.conv5_2(conv5), 0.2)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv2], 1)
        conv6 = F.leaky_relu(self.conv6_1(up6), 0.2)
        conv6 = F.leaky_relu(self.conv6_2(conv6), 0.2)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv1], 1)
        conv7 = F.leaky_relu(self.conv7_1(up7), 0.2)
        conv7 = F.leaky_relu(self.conv7_2(conv7), 0.2)

        conv8 = self.conv8_1(conv7)
        out = conv8

        return out
