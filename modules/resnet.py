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


# ============================================================
# Global Config: switch scheme by modifying this dict
# Options:
# "static", "condconv", "dyconv", "odconv",
# "tada_v1", "tada_v2", "giconv2d",
# "se", "skconv", "cw_condconv", "rank_adaptive"
# ============================================================

DYNAMIC_FUSION = {
    "type": "skconv",
    "num_experts": 4,
    "reduction": 8,
    "dy_temperature": 30.0,
    "gi_blocks": 2,
    "sk_branches": 3,
}


def set_dynamic_fusion_type(name):
    """
    Must be called before model = resnet18(...) / resnet34(...).
    Changing this global variable after model instantiation will not affect
    modules that have already been created.
    """
    valid = {
        "static", "condconv", "dyconv", "dynamicconv", "odconv",
        "tada_v1", "tada_v2", "giconv2d",
        "se", "skconv", "cw_condconv", "rank_adaptive"
    }
    name = name.lower()
    if name not in valid:
        raise ValueError(f"Unknown fusion type: {name}. Valid types: {valid}")
    DYNAMIC_FUSION["type"] = name


def _hidden_channels(c, reduction=8):
    return max(1, c // reduction)


def _to_3tuple(k):
    if isinstance(k, tuple):
        return k
    return (k, k, k)


def _apply_per_sample_dwconv3d(x, weight, stride=1, padding=0, dilation=1):
    """
    x:      [B, C, T, H, W]
    weight: [B, C, 1, kT, kH, kW]
    Applies a distinct depthwise conv3d kernel per sample and per channel.
    """
    B, C, T, H, W = x.shape
    kT, kH, kW = weight.shape[-3:]

    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    x_g = x.reshape(1, B * C, T, H, W)
    w_g = weight.reshape(B * C, 1, kT, kH, kW)

    y = F.conv3d(
        x_g,
        w_g,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=B * C
    )

    _, _, To, Ho, Wo = y.shape
    return y.reshape(B, C, To, Ho, Wo)


def _apply_per_frame_dwconv2d(x, weight, alpha, stride=1, padding=0, dilation=1):
    """
    Per-frame spatial kernel calibration in the style of TAda / GIConv2d.

    x:      [B, C, T, H, W]
    weight: [C, 1, kH, kW]
    alpha:  [B, C, T]
    """
    B, C, T, H, W = x.shape
    kH, kW = weight.shape[-2:]

    if isinstance(stride, tuple):
        stride = stride[-1]
    if isinstance(padding, tuple):
        padding = padding[-1]
    if isinstance(dilation, tuple):
        dilation = dilation[-1]

    x_g = x.permute(0, 2, 1, 3, 4).contiguous()
    x_g = x_g.reshape(1, B * T * C, H, W)

    a = alpha.permute(0, 2, 1).contiguous().view(B, T, C, 1, 1, 1)
    w = weight.view(1, 1, C, 1, kH, kW) * a
    w_g = w.reshape(B * T * C, 1, kH, kW)

    y = F.conv2d(
        x_g,
        w_g,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=B * T * C
    )

    _, _, Ho, Wo = y.shape
    y = y.reshape(B, T, C, Ho, Wo)
    return y.permute(0, 2, 1, 3, 4).contiguous()


class StaticDWConv3d(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# 1. CondConv-style depthwise Conv3d
# ============================================================

class CondConvDWConv3d(nn.Module):
    """
    CondConv idea:
    Generate routing weights for each sample, then linearly combine
    multiple expert kernels.
    """
    def __init__(self, channels, kernel_size, padding, num_experts=4, reduction=8):
        super().__init__()
        self.channels = channels
        self.kernel_size = _to_3tuple(kernel_size)
        self.padding = _to_3tuple(padding)
        self.num_experts = num_experts

        hidden = _hidden_channels(channels, reduction)

        self.weight = nn.Parameter(
            torch.empty(num_experts, channels, 1, *self.kernel_size)
        )

        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_normal_(
                self.weight[i],
                mode="fan_out",
                nonlinearity="relu"
            )

    def forward(self, x):
        B = x.size(0)
        route = self.routing(x).view(B, self.num_experts, 1, 1, 1, 1, 1)
        weight = (route * self.weight.unsqueeze(0)).sum(dim=1)
        return _apply_per_sample_dwconv3d(x, weight, padding=self.padding)


# ============================================================
# 2. DyConv-style depthwise Conv3d
# ============================================================

class DyConvDWConv3d(nn.Module):
    """
    DyConv / DynamicConv idea:
    Dynamically aggregate multiple convolution kernels using softmax attention.
    """
    def __init__(
        self,
        channels,
        kernel_size,
        padding,
        num_experts=4,
        reduction=8,
        temperature=30.0
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = _to_3tuple(kernel_size)
        self.padding = _to_3tuple(padding)
        self.num_experts = num_experts
        self.temperature = temperature

        hidden = _hidden_channels(channels, reduction)

        self.weight = nn.Parameter(
            torch.empty(num_experts, channels, 1, *self.kernel_size)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_normal_(
                self.weight[i],
                mode="fan_out",
                nonlinearity="relu"
            )

    def forward(self, x):
        B = x.size(0)
        att = self.attention(x) / self.temperature
        att = F.softmax(att, dim=1).view(B, self.num_experts, 1, 1, 1, 1, 1)

        weight = (att * self.weight.unsqueeze(0)).sum(dim=1)
        return _apply_per_sample_dwconv3d(x, weight, padding=self.padding)


# ============================================================
# 3. ODConv-style depthwise Conv3d
# ============================================================

class ODConvDWConv3d(nn.Module):
    """
    ODConv idea:
    Not only applies attention over kernel numbers, but also over
    input-channel, output-filter, and spatial kernel positions.
    Light-weight adaptation for depthwise Conv3d.
    """
    def __init__(self, channels, kernel_size, padding, num_experts=4, reduction=8):
        super().__init__()
        self.channels = channels
        self.kernel_size = _to_3tuple(kernel_size)
        self.padding = _to_3tuple(padding)
        self.num_experts = num_experts

        kT, kH, kW = self.kernel_size
        hidden = _hidden_channels(channels, reduction)

        self.weight = nn.Parameter(
            torch.empty(num_experts, channels, 1, kT, kH, kW)
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
        )

        self.kernel_fc = nn.Linear(hidden, num_experts)
        self.in_channel_fc = nn.Linear(hidden, channels)
        self.out_filter_fc = nn.Linear(hidden, channels)
        self.spatial_fc = nn.Linear(hidden, kT * kH * kW)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_normal_(
                self.weight[i],
                mode="fan_out",
                nonlinearity="relu"
            )

    def forward(self, x):
        B, C, T, H, W = x.shape
        kT, kH, kW = self.kernel_size

        z = self.shared(self.pool(x))

        kernel_att = F.softmax(self.kernel_fc(z), dim=1)
        kernel_att = kernel_att.view(B, self.num_experts, 1, 1, 1, 1, 1)

        in_att = torch.sigmoid(self.in_channel_fc(z)).view(B, C, 1, 1, 1)
        out_att = torch.sigmoid(self.out_filter_fc(z)).view(B, C, 1, 1, 1)

        spatial_att = torch.sigmoid(self.spatial_fc(z))
        spatial_att = spatial_att.view(B, 1, 1, 1, kT, kH, kW)

        x = x * in_att

        weight = (
            self.weight.unsqueeze(0) * kernel_att * spatial_att
        ).sum(dim=1)

        y = _apply_per_sample_dwconv3d(x, weight, padding=self.padding)
        y = y * out_att

        return y


# ============================================================
# 4. TAda V1-style Conv2d over time
# ============================================================

class TAdaV1DWConv2dOverTime(nn.Module):
    """
    TAda V1 idea:
    Generate spatial kernel calibration coefficients for each frame
    based on its temporal descriptor.
    Designed for spatial branch with kernel_size=(1,k,k).
    """
    def __init__(self, channels, kernel_size, padding, reduction=8):
        super().__init__()

        kT, kH, kW = _to_3tuple(kernel_size)
        pT, pH, pW = _to_3tuple(padding)

        assert kT == 1, "TAdaV1DWConv2dOverTime only supports kernel_size=(1,k,k)"

        self.padding = pH
        self.weight = nn.Parameter(torch.empty(channels, 1, kH, kW))

        hidden = _hidden_channels(channels, reduction)

        self.route = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=3, padding=1, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_out",
            nonlinearity="relu"
        )

    def forward(self, x):
        # [B,C,T,H,W] -> [B,C,T]
        desc = x.mean(dim=(3, 4))

        # 2 * sigmoid keeps calibration factor close to 1 at initialization
        alpha = 2.0 * torch.sigmoid(self.route(desc))

        return _apply_per_frame_dwconv2d(
            x,
            self.weight,
            alpha,
            padding=self.padding
        )


# ============================================================
# 5. TAda V2-style Conv2d over time
# ============================================================

class TAdaV2DWConv2dOverTime(nn.Module):
    """
    TAda V2 idea:
    Builds on per-frame local temporal descriptors and introduces
    global temporal context.
    Light-weight engineering implementation, not an official line-by-line reproduction.
    """
    def __init__(self, channels, kernel_size, padding, reduction=8):
        super().__init__()

        kT, kH, kW = _to_3tuple(kernel_size)
        pT, pH, pW = _to_3tuple(padding)

        assert kT == 1, "TAdaV2DWConv2dOverTime only supports kernel_size=(1,k,k)"

        self.padding = pH
        self.weight = nn.Parameter(torch.empty(channels, 1, kH, kW))

        hidden = _hidden_channels(channels, reduction)

        self.local = nn.Conv1d(
            channels,
            hidden,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.global_proj = nn.Conv1d(
            channels,
            hidden,
            kernel_size=1,
            bias=True
        )

        self.out = nn.Conv1d(
            hidden,
            channels,
            kernel_size=1,
            bias=True
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_out",
            nonlinearity="relu"
        )

    def forward(self, x):
        desc = x.mean(dim=(3, 4))  # [B,C,T]

        global_ctx = desc.mean(dim=2, keepdim=True).expand_as(desc)

        h = self.local(desc) + self.global_proj(global_ctx)
        h = F.relu(h, inplace=True)

        alpha = 2.0 * torch.sigmoid(self.out(h))

        return _apply_per_frame_dwconv2d(
            x,
            self.weight,
            alpha,
            padding=self.padding
        )


# ============================================================
# 6. GIConv2d-style Conv2d over time
# ============================================================

def _valid_num_heads(channels, max_heads=8):
    for h in range(min(max_heads, channels), 0, -1):
        if channels % h == 0:
            return h
    return 1


class GIBlock(nn.Module):
    """
    Global Interaction Block:
    Applies temporal self-attention to frame-level global descriptors [B,T,C].
    """
    def __init__(self, channels, mlp_ratio=2.0, max_heads=8):
        super().__init__()

        heads = _valid_num_heads(channels, max_heads)
        hidden = int(channels * mlp_ratio)

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels,
            heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels)
        )

    def forward(self, z):
        h = self.norm1(z)
        h, _ = self.attn(h, h, h, need_weights=False)
        z = z + h

        z = z + self.ffn(self.norm2(z))
        return z


class GIConv2dOverTime(nn.Module):
    """
    GIConv2d idea:
    First models global interactions among frame descriptors via GIBlock,
    then generates per-frame convolution kernel calibration weights.
    """
    def __init__(self, channels, kernel_size, padding, reduction=8, gi_blocks=2):
        super().__init__()

        kT, kH, kW = _to_3tuple(kernel_size)
        pT, pH, pW = _to_3tuple(padding)

        assert kT == 1, "GIConv2dOverTime only supports kernel_size=(1,k,k)"

        self.padding = pH
        self.weight = nn.Parameter(torch.empty(channels, 1, kH, kW))

        self.gi = nn.Sequential(
            *[GIBlock(channels) for _ in range(gi_blocks)]
        )

        hidden = _hidden_channels(channels, reduction)

        self.calibrator = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_out",
            nonlinearity="relu"
        )

    def forward(self, x):
        # x: [B,C,T,H,W]
        # desc: [B,T,C]
        desc = x.mean(dim=(3, 4)).transpose(1, 2).contiguous()

        z = self.gi(desc)

        # [B,T,C] -> [B,C,T]
        alpha = self.calibrator(z).transpose(1, 2).contiguous()
        alpha = 2.0 * torch.sigmoid(alpha)

        return _apply_per_frame_dwconv2d(
            x,
            self.weight,
            alpha,
            padding=self.padding
        )


# ============================================================
# 7. SE-style dynamic calibration depthwise Conv3d
# ============================================================

class SEDWConv3d(nn.Module):
    """
    SE dynamic calibration idea:
    The convolution kernel remains static, but a channel-wise gate is
    generated from the current sample to dynamically rescale the
    depthwise Conv3d output.

    Pros: extremely light-weight, almost zero inference overhead;
    suitable as a baseline or small-dataset overfitting prevention.
    """
    def __init__(self, channels, kernel_size, padding, reduction=8):
        super().__init__()
        self.conv = StaticDWConv3d(channels, kernel_size, padding)
        hidden = _hidden_channels(channels, reduction)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        # 2 * sigmoid keeps gate close to 1 at initialization to avoid
        # excessive suppression during early training.
        g = 2.0 * self.gate(x).view(x.size(0), x.size(1), 1, 1, 1)
        return y * g


# ============================================================
# 8. SKConv-style selective-kernel depthwise Conv3d
# ============================================================

def _grow_kernel_size(kernel_size, grow):
    """
    Only expands non-1 dimensions:
    temporal branch: (3,1,1) -> (3,1,1), (5,1,1), (7,1,1)
    spatial branch:  (1,3,3) -> (1,3,3), (1,5,5), (1,7,7)
    """
    kT, kH, kW = _to_3tuple(kernel_size)
    return (
        kT if kT == 1 else kT + 2 * grow,
        kH if kH == 1 else kH + 2 * grow,
        kW if kW == 1 else kW + 2 * grow,
    )


def _same_padding_3d(kernel_size):
    kT, kH, kW = _to_3tuple(kernel_size)
    return (kT // 2, kH // 2, kW // 2)


class SKDWConv3d(nn.Module):
    """
    Selective Kernel idea:
    Parallel multiple depthwise Conv3d branches with different receptive fields,
    then adaptively generate channel-wise softmax weights for each branch
    based on the input.

    Difference from CondConv / DyConv:
    - CondConv / DyConv dynamically aggregate multiple kernels;
    - SKConv dynamically selects among multiple scale branch outputs.
    """
    def __init__(self, channels, kernel_size, padding, branches=3, reduction=8):
        super().__init__()
        self.channels = channels
        self.branches = branches

        self.convs = nn.ModuleList()
        for i in range(branches):
            k = _grow_kernel_size(kernel_size, i)
            p = _same_padding_3d(k)
            self.convs.append(
                nn.Sequential(
                    nn.Conv3d(
                        channels,
                        channels,
                        kernel_size=k,
                        padding=p,
                        groups=channels,
                        bias=False
                    ),
                    nn.BatchNorm3d(channels),
                    nn.ReLU(inplace=True)
                )
            )

        hidden = _hidden_channels(channels, reduction)
        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fcs = nn.ModuleList([
            nn.Conv3d(hidden, channels, 1, bias=True)
            for _ in range(branches)
        ])

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        u = sum(feats)
        z = self.shared(u)

        # [B, M, C, 1, 1, 1]
        att = torch.stack([fc(z) for fc in self.fcs], dim=1)
        att = F.softmax(att, dim=1)

        y = 0
        for i, feat in enumerate(feats):
            y = y + feat * att[:, i]
        return y


# ============================================================
# 9. Channel-wise CondConv depthwise Conv3d
# ============================================================

class ChannelWiseCondConvDWConv3d(nn.Module):
    """
    Channel-wise CondConv idea:
    Standard CondConv generates global num_experts routing weights per sample;
    here we further generate expert weights independently for each channel.

    Well-suited for depthwise conv since each channel is already independent.
    Usually more expressive than standard CondConv, yet still lighter than
    full dynamic convolution.
    """
    def __init__(self, channels, kernel_size, padding, num_experts=4, reduction=8):
        super().__init__()
        self.channels = channels
        self.kernel_size = _to_3tuple(kernel_size)
        self.padding = _to_3tuple(padding)
        self.num_experts = num_experts

        hidden = _hidden_channels(channels, reduction)

        self.weight = nn.Parameter(
            torch.empty(num_experts, channels, 1, *self.kernel_size)
        )

        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * num_experts)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_normal_(
                self.weight[i],
                mode="fan_out",
                nonlinearity="relu"
            )

    def forward(self, x):
        B, C, T, H, W = x.shape

        # [B, C, E] -> [B, E, C, 1, 1, 1, 1]
        route = self.routing(x).view(B, C, self.num_experts)
        route = F.softmax(route, dim=2).permute(0, 2, 1).contiguous()
        route = route.view(B, self.num_experts, C, 1, 1, 1, 1)

        weight = (route * self.weight.unsqueeze(0)).sum(dim=1)
        return _apply_per_sample_dwconv3d(x, weight, padding=self.padding)


# ============================================================
# 10. Low-rank rank-adaptive dynamic depthwise Conv3d
# ============================================================

class RankAdaptiveDWConv3d(nn.Module):
    """
    Low-rank dynamic kernel idea:
    Instead of generating a full [C,kT,kH,kW] dynamic kernel,
    decompose dynamics into a channel gate and a kernel-position gate:

        W_dynamic = W_base * A_channel(x) * A_kernel_position(x)

    Lighter than ODConv; closer to true dynamic convolution than SE.
    """
    def __init__(self, channels, kernel_size, padding, reduction=8):
        super().__init__()
        self.channels = channels
        self.kernel_size = _to_3tuple(kernel_size)
        self.padding = _to_3tuple(padding)

        kT, kH, kW = self.kernel_size
        hidden = _hidden_channels(channels, reduction)

        self.base_weight = nn.Parameter(
            torch.empty(channels, 1, kT, kH, kW)
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True)
        )

        self.channel_fc = nn.Linear(hidden, channels)
        self.kernel_fc = nn.Linear(hidden, kT * kH * kW)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.base_weight,
            mode="fan_out",
            nonlinearity="relu"
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        kT, kH, kW = self.kernel_size

        z = self.shared(self.pool(x))

        channel_gate = 2.0 * torch.sigmoid(self.channel_fc(z))
        channel_gate = channel_gate.view(B, C, 1, 1, 1, 1)

        kernel_gate = 2.0 * torch.sigmoid(self.kernel_fc(z))
        kernel_gate = kernel_gate.view(B, 1, 1, kT, kH, kW)

        weight = self.base_weight.unsqueeze(0) * channel_gate * kernel_gate
        return _apply_per_sample_dwconv3d(x, weight, padding=self.padding)


# ============================================================
# Factory
# ============================================================

def build_dynamic_dwconv3d(channels, kernel_size, padding, branch_type="temporal"):
    mode = DYNAMIC_FUSION["type"].lower()
    n = DYNAMIC_FUSION.get("num_experts", 4)
    r = DYNAMIC_FUSION.get("reduction", 8)

    if mode in ("none", "static"):
        return StaticDWConv3d(channels, kernel_size, padding)

    if mode == "condconv":
        return CondConvDWConv3d(
            channels,
            kernel_size,
            padding,
            num_experts=n,
            reduction=r
        )

    if mode in ("dyconv", "dynamicconv"):
        return DyConvDWConv3d(
            channels,
            kernel_size,
            padding,
            num_experts=n,
            reduction=r,
            temperature=DYNAMIC_FUSION.get("dy_temperature", 30.0)
        )

    if mode == "odconv":
        return ODConvDWConv3d(
            channels,
            kernel_size,
            padding,
            num_experts=n,
            reduction=r
        )

    if mode == "se":
        return SEDWConv3d(
            channels,
            kernel_size,
            padding,
            reduction=r
        )

    if mode == "skconv":
        return SKDWConv3d(
            channels,
            kernel_size,
            padding,
            branches=DYNAMIC_FUSION.get("sk_branches", 3),
            reduction=r
        )

    if mode == "cw_condconv":
        return ChannelWiseCondConvDWConv3d(
            channels,
            kernel_size,
            padding,
            num_experts=n,
            reduction=r
        )

    if mode == "rank_adaptive":
        return RankAdaptiveDWConv3d(
            channels,
            kernel_size,
            padding,
            reduction=r
        )

    if mode == "tada_v1":
        if branch_type == "spatial":
            return TAdaV1DWConv2dOverTime(
                channels,
                kernel_size,
                padding,
                reduction=r
            )
        return StaticDWConv3d(channels, kernel_size, padding)

    if mode == "tada_v2":
        if branch_type == "spatial":
            return TAdaV2DWConv2dOverTime(
                channels,
                kernel_size,
                padding,
                reduction=r
            )
        return StaticDWConv3d(channels, kernel_size, padding)

    if mode == "giconv2d":
        if branch_type == "spatial":
            return GIConv2dOverTime(
                channels,
                kernel_size,
                padding,
                reduction=r,
                gi_blocks=DYNAMIC_FUSION.get("gi_blocks", 2)
            )
        return StaticDWConv3d(channels, kernel_size, padding)

    raise ValueError(f"Unknown DYNAMIC_FUSION['type']: {mode}")


# ============================================================
# Replace your original SpatioTemporalFusion
# ============================================================

class SpatioTemporalFusion(nn.Module):
    def __init__(self, in_channels, n_segment=8):
        super().__init__()

        self.in_channels = in_channels
        self.n_segment = n_segment
        self.mode = DYNAMIC_FUSION["type"]

        # Multi-scale temporal convolution
        # CondConv / DyConv / ODConv will dynamize this branch;
        # TAda / GIConv2d mainly act on spatial conv, so temporal stays static.
        self.t_convs = nn.ModuleList([
            nn.Sequential(
                build_dynamic_dwconv3d(
                    in_channels,
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0),
                    branch_type="temporal"
                ),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                build_dynamic_dwconv3d(
                    in_channels,
                    kernel_size=(5, 1, 1),
                    padding=(2, 0, 0),
                    branch_type="temporal"
                ),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            )
        ])

        # Multi-scale spatial convolution
        self.s_convs = nn.ModuleList([
            nn.Sequential(
                build_dynamic_dwconv3d(
                    in_channels,
                    kernel_size=(1, 3, 3),
                    padding=(0, 1, 1),
                    branch_type="spatial"
                ),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                build_dynamic_dwconv3d(
                    in_channels,
                    kernel_size=(1, 5, 5),
                    padding=(0, 2, 2),
                    branch_type="spatial"
                ),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            )
        ])

        # Dynamic fusion weights for 4 branches
        hidden = _hidden_channels(in_channels, 8)

        self.calibration = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 4, 1),
            nn.Softmax(dim=1)
        )

        # Spatio-temporal interaction gate
        gate_hidden = max(1, in_channels // 2)

        self.st_gate = nn.Sequential(
            nn.Conv3d(in_channels * 2, gate_hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(gate_hidden, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,C,T,H,W]

        t_feats = [t_conv(x) for t_conv in self.t_convs]
        s_feats = [s_conv(x) for s_conv in self.s_convs]

        # [B,4,C,T,H,W]
        all_feats = torch.stack(t_feats + s_feats, dim=1)

        # [B,4,1,1,1]
        weights = self.calibration(x)

        # [B,C,T,H,W]
        fused = (all_feats * weights.unsqueeze(2)).sum(dim=1)

        gate = self.st_gate(torch.cat([x, fused], dim=1))

        # Returns the residual branch.
        # Your ResNet already has: x = x + self.shift(x) * alpha
        return gate * fused


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False
    )


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
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
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
                          kernel_size=1, stride=(1, stride, stride), bias=False),
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

        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])  # bt, c, h, w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # bt, c
        x = self.fc(x)  # bt, c

        return x, res


def resnet18(**kwargs):
    """Constructs a ResNet-18 based model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x