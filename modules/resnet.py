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
# ------------------------------------------------------------------
# Spatio-Temporal Fusion Module
# Fuse temporal and spatial features via multi-scale 3-D convolutions
# ------------------------------------------------------------------
class SpatioTemporalFusion(nn.Module):
    def __init__(self, in_channels, n_segment=8):
        super().__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment

        # Temporal convolutions: (kernel_size) = (3,1,1) & (5,1,1)
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

        # Spatial convolutions: (kernel_size) = (1,3,3) & (1,5,5)
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

        # Calibration branch: learn 4 weights for the four paths above
        self.calibration = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 8, 4, 1),  # 4-branch weights
            nn.Softmax(dim=1)
        )

        # Spatial-temporal gate to re-weight the fused features
        self.st_gate = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        nt, c, t, h, w = x.size()

        # Extract multi-scale temporal features
        t_feats = [t_conv(x) for t_conv in self.t_convs]

        # Extract multi-scale spatial features
        s_feats = [s_conv(x) for s_conv in self.s_convs]

        # Concatenate all feature maps
        all_feats = t_feats + s_feats
        all_feats = torch.stack(all_feats, dim=1)  # [B, 4, C, T, H, W]

        # Learn 4 weights via calibration branch
        weights = self.calibration(x)  # [B, 4, 1, 1, 1]
        fused = (all_feats * weights.unsqueeze(2)).sum(dim=1)

        # Gate mechanism to refine residual
        gate = self.st_gate(torch.cat([x, fused], dim=1))
        output = x + gate * fused

        return output

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
        self.stf1 = SpatioTemporalFusion(128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.stf2 = SpatioTemporalFusion(256)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.stf3 = SpatioTemporalFusion(512)
        self.avgpool = nn.AvgPool2d(7, stride=1)
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
        x = x + self.stf1(x) * self.alpha[0]

        x = self.layer3(x)
        res.append(x)
        x = x + self.stf2(x) * self.alpha[1]

        x = self.layer4(x)
        res.append(x)
        x = x + self.stf3(x) * self.alpha[2]

        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:]) #bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #bt,c
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

# ------------------------------------------------------------------
# Spatio-Temporal Squeeze-and-Excitation Module (STSE)
# Combines channel attention & temporal attention
# ------------------------------------------------------------------
class SpatioTemporalSE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatioTemporalSE, self).__init__()
        # Global average & max pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Channel attention
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # Temporal attention via 1-D temporal convolutions
        self.temporal_att = nn.Sequential(
            nn.Conv3d(channel, channel // 4, kernel_size=(3, 1, 1),
                      padding=(1, 0, 0), groups=channel // 16),
            nn.BatchNorm3d(channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // 4, channel, kernel_size=(3, 1, 1),
                      padding=(1, 0, 0), groups=channel // 16),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        # Channel attention
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = self.fc(y_avg) + self.fc(y_max)
        channel_att = y.view(b, c, 1, 1, 1)

        # Temporal attention
        temporal_att = self.temporal_att(x)

        # Combine and apply
        att = torch.sigmoid(channel_att + temporal_att)
        return x * att.expand_as(x)


# ------------------------------------------------------------------
# Hierarchical Feature Extraction & Aggregation Module (HFEAM)
# Aggregates multi-scale features via STSE + TCN
# ------------------------------------------------------------------
class HFEAM(nn.Module):
    def __init__(self, inplanes, planes=1024):
        super(HFEAM, self).__init__()

        # STSE modules for multi-scale features
        self.stse1 = SpatioTemporalSE(512)
        self.stse2 = SpatioTemporalSE(512)
        self.stse3 = SpatioTemporalSE(512)

        # Projection layers to unify channel dimension to 512
        self.con1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.Conv3d(64, 512, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.con2 = nn.Sequential(
            nn.Conv3d(128, 512, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        self.con3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        # Temporal Convolutional Network (TCN) to capture temporal dynamics
        self.tcn_layers = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout to prevent overfitting
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
            nn.Conv1d(planes, planes, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
        )

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, 1000)

    def forward(self, res):
        # res: list of multi-scale features from ResNet
        # res[0]: stage1, res[1]: stage2, res[2]: stage3, res[3]: stage4
        B, _, T, _, _ = res[1].shape
        multi = []

        # Project & apply STSE for each scale
        multi.append(self.stse1(self.con2(res[1])))
        multi.append(self.stse2(self.con3(res[2])))
        multi.append(self.stse3(res[3]))

        # Flatten spatial dimensions & apply TCN
        for i in range(3):
            # (B,C,T,H,W) -> (B*T,C,H,W)
            multi[i] = multi[i].transpose(1, 2).contiguous()
            multi[i] = multi[i].view((-1,) + multi[i].size()[2:])

            # Global average pooling
            multi[i] = self.avgpool(multi[i])
            multi[i] = multi[i].view(multi[i].size(0), -1)

            # FC to get per-frame logits
            multi[i] = self.fc(multi[i])

            # Reshape back to (B,T,C) then (B,C,T) for TCN
            multi[i] = multi[i].view(B, T, -1).permute(0, 2, 1)

            # Apply TCN to capture temporal dependencies
            multi[i] = self.tcn_layers(multi[i])

        return multi


# ------------------------------------------------------------------
# Demo script
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time
    start_time = time.time()

    # Build ResNet-18 3-D backbone
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.fc = Identity()  # Remove final FC layer

    # Dummy input: (B, C, T, H, W)
    batch, temp = 4, 16
    video_input = torch.randn(batch, 3, temp, 224, 224)
    output = model(video_input)

    # output[0]: final logits (B*T, C); output[1]: list of intermediate features
    print("Output shape (logits):", output[0].shape)

    # Print shapes of all intermediate feature maps
    for i in range(4):
        print(f"Shape of ResNet stage {i} output: {output[1][i].shape}")

    # Build HFEAM module
    HFEAM = HFEAM(1000)
    res2 = HFEAM(output[1])

    # Print shapes of HFEAM outputs
    for i in range(3):
        print(f"Shape of HFEAM branch {i} output: {res2[i].shape}")

    end_time = time.time()
    print("Total inference time:", end_time - start_time)