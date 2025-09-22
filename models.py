import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.Hardtanh(min_val=-1.0, max_val=1.0)(x) # 修改处
        out = F.relu(self.bn1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # [64, 16, 16]
        # print(f"weight = {self.conv1.weight.data.shape}")
        self.features = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [64, 8, 8]
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # [192, 8, 8]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [192, 4, 4]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # [384, 4, 4]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # [256, 4, 4]
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # [256, 4, 4]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [256, 2, 2]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = nn.Hardtanh(min_val=-1.0, max_val=1.0)(x)  # 修改处
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平 [B, C, H, W] -> [B, C*H*W]
        x = self.classifier(x)
        return x

class VGGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.features = nn.Sequential(
            # Block 1: 2 conv + maxpool
            nn.BatchNorm2d(64),  # 添加 BatchNorm 加速收敛
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 16, 16]
            # Block 2: 2 conv + maxpool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128, 8, 8]
            # Block 3: 2 conv + maxpool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [256, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = nn.Hardtanh(min_val=-1.0, max_val=1.0)(x)  # 修改处
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平为 [B, 256*4*4]
        x = self.classifier(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5_mid, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()

        # Branch 1: 1x1 conv
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # Branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )

        # Branch 3: 1x1 -> 3x3 -> 3x3 (replacing 5x5)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5_mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_mid, ch5x5, kernel_size=3, padding=1)
        )

        # Branch 4: MaxPool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))
        branch2 = F.relu(self.branch2(x))
        branch3 = F.relu(self.branch3(x))
        branch4 = F.relu(self.branch4(x))
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# Define the GoogLeNet architecture adapted for CIFAR-10 (32x32 images).
# We adjust the initial convolutions and pooling to fit smaller input size.
class GoogLeNet3x3(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet3x3, self).__init__()

        # Initial layers adjusted for 32x32 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Changed from 7x7 to 3x3
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception modules
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128, 128)

        # Global average pooling and FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x) # 修改处
        x = nn.Hardtanh(min_val=-1.0, max_val=1.0)(x)  # 修改处
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet20()
    print(f"model.conv1.weight.shape = {model.conv1.weight.shape}")
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, 10]