import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}




class SpatioTemporalFusion(nn.Module):
    def __init__(self, in_channels, n_segment=8):
        super().__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment

        # 多尺度时间卷积 (3D)
        self.t_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, (3, 1, 1),
                          padding=(1, 0, 0), groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, (5, 1, 1),
                          padding=(2, 0, 0), groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            )
        ])

        # 多尺度空间卷积 (2D)
        self.s_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, (1, 3, 3),
                          padding=(0, 1, 1), groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, (1, 5, 5),
                          padding=(0, 2, 2), groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            )
        ])

        # 动态特征校准
        self.calibration = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 8, 4, 1),  # 4个分支的权重
            nn.Softmax(dim=1)
        )

        # 时空交互门
        self.st_gate = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        nt, c, t, h, w = x.size()

        # 时间特征分支
        t_feats = [t_conv(x) for t_conv in self.t_convs]

        # 空间特征分支
        s_feats = [s_conv(x) for s_conv in self.s_convs]

        # 特征融合
        all_feats = t_feats + s_feats
        all_feats = torch.stack(all_feats, dim=1)  # [B, 4, C, T, H, W]

        # 动态权重校准
        weights = self.calibration(x)  # [B, 4, 1, 1, 1]
        fused = (all_feats * weights.unsqueeze(2)).sum(dim=1)

        # 时空交互门控
        gate = self.st_gate(torch.cat([x, fused], dim=1))
        output = x + gate * fused

        return gate * fused

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)



        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.shift1 = SpatioTemporalFusion(128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.shift2 = SpatioTemporalFusion(256)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.shift3 = SpatioTemporalFusion(512)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        res = []
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        res.append(x)
        x = self.layer2(x)
        res.append(x)
        x = x + self.shift1(x) * self.alpha[0]

        x = self.layer3(x)
        res.append(x)
        x = x + self.shift2(x) * self.alpha[1]

        x = self.layer4(x)
        res.append(x)
        x = x + self.shift3(x) * self.alpha[2]

        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:]) #bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #bt,c
        # x = self.dropout(x)
        x = self.fc(x) #bt,c

        return x,res



def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


