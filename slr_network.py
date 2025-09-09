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

        self.HFEAM = HFEAM(1000)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier


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
            framewise,res = self.conv2d(x.permute(0,2,1,3,4)) # btc -> bct
            framewise = framewise.view(batch, temp, -1).permute(0,2,1)
        else:
            # frame-wise features
            framewise = x

        res2 = self.HFEAM(res)
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
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "res2":              res2,
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

        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['mse_loss'] = nn.MSELoss()
        return self.loss
