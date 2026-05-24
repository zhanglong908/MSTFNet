import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class MTD(nn.Module):
    #Multi-Scale Temporal Downsample Block, MTD 多尺度时序降采样模块
    def __init__(self, hidden_size):
        super(MTD, self).__init__()
        self.dw_k3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size, bias=False)
        self.dw_k5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size, bias=False)

        self.shrink_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, stride=1, padding=0, groups=hidden_size,
                                     bias=False)

        self.pointwise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.act = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=False)

    def forward(self, x):
        x_multi = x + self.dw_k3(x) + self.dw_k5(x)

        x_shrink = self.shrink_conv(x_multi)

        out = self.pointwise(x_shrink)
        out = self.bn(out)
        out = self.act(out)

        # 4. 原汁原味的 MaxPool 保证尖峰传导
        return self.pool(out)

class SpatioTemporalSE(nn.Module):
    """时空SE模块，适用于视频数据"""

    def __init__(self, channel, reduction=16):
        super(SpatioTemporalSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # # 双路径融合机制
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )

        # 时间注意力分支
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

        # # 通道注意力
        # y_avg = self.avg_pool(x).view(b, c)
        # y_max = self.max_pool(x).view(b, c)
        # y = self.fc(y_avg) + self.fc(y_max)
        # channel_att = y.view(b, c, 1, 1, 1)

        # 时间注意力
        temporal_att = self.temporal_att(x)

        # 融合注意力
        # att = torch.sigmoid(channel_att + temporal_att)
        return x * temporal_att.expand_as(x)

class CSA(nn.Module):
    def __init__(self, inplanes, planes=1024):   # inplanes 保留接口但不使用
        super(CSA, self).__init__()
        self.stf1 = SpatioTemporalSE(512)
        self.stf2 = SpatioTemporalSE(512)
        self.stf3 = SpatioTemporalSE(512)

        # 保持原有的下采样卷积
        self.con2 = nn.Conv3d(128, 512, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.con3 = nn.Conv3d(256, 512, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 替换原 tcn_layers 为两个 MTD 堆叠（保证与原两次下采样一致）
        self.tcn_layers = nn.Sequential(
            MTD(planes),
            MTD(planes)
        )

        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 调整 fc 输出为 planes，使通道与 MTD 匹配
        self.fc = nn.Linear(512, planes)

    def forward(self, res):
        # res[1] -> layer2, res[2] -> layer3, res[3] -> layer4
        B, _, T, _, _ = res[1].shape
        multi = []

        feat2 = self.stf1(self.con2(res[1]))
        feat3 = self.stf2(self.con3(res[2]))
        feat4 = self.stf3(res[3])

        for feat in [feat2, feat3, feat4]:
            feat = feat.transpose(1, 2).contiguous()
            feat = feat.view((-1,) + feat.size()[2:])
            feat = self.avgpool(feat)
            feat = feat.view(feat.size(0), -1)
            feat = self.fc(feat)               # 输出维度 planes (1024)
            feat = feat.view(B, T, -1).permute(0, 2, 1)  # (B, planes, T)
            feat = self.tcn_layers(feat)       # 两个 MTD，输出 (B, planes, T')
            multi.append(feat)

        return multi
class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)

        self.csa = CSA(1000)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * 各版本的resnet原版, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, 各版本的resnet原版, -1).transpose(1, 2)
            framewise,res = self.conv2d(x.permute(0,2,1,3,4)) # btc -> bct
            framewise = framewise.view(batch, temp, -1).permute(0,2,1)
        else:
            # frame-wise features
            framewise = x

        res2 = self.csa(res)                # 正确：传入 layer2, layer3, layer4
        for i in range(3):
            res2[i] = res2[i].permute(2,0,1)
            res2[i] = self.classifier(res2[i])  



        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            #"visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "res2":              res2,
            "loss_LiftPool_u": conv1d_outputs['loss_LiftPool_u'],
            "loss_LiftPool_p": conv1d_outputs['loss_LiftPool_p'],
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()

                loss +=  self.loss['CTCLoss'](ret_dict["res2"][0].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()    
                loss +=  self.loss['CTCLoss'](ret_dict["res2"][1].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean() 
                loss +=  self.loss['CTCLoss'](ret_dict["res2"][2].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()    
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
            elif k == 'Cu':
                loss += weight * ret_dict["loss_LiftPool_u"]
            elif k == 'Cp':
                loss += weight * ret_dict["loss_LiftPool_p"]

        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['mse_loss'] = nn.MSELoss()
        return self.loss
