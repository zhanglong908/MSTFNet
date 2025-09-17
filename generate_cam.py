import numpy as np
import os
import glob
import cv2
from utils import video_augmentation
from slr_network import SLRModel
import torch
from collections import OrderedDict
import utils

gpu_id = 0  # which gpu to use
dataset = 'phoenix2014'  # support [phoenix2014, phoenix2014-T, CSL-Daily]
prefix = './dataset/phoenix2014/phoenix-2014-multisigner'  # ['./dataset/CSL-Daily', './dataset/phoenix2014-T', './dataset/phoenix2014/phoenix-2014-multisigner']
dict_path = f'./preprocess/{dataset}/gloss_dict.npy'
model_weights = '/remote-home/cs_cs_zl/SpatioTemporalSE/154SpatioTemporalSE/154SpatioTemporalSEdev_17.10_epoch63_model.pt'
select_id = 100

fmap_block = []
grad_block = []


def forward_hook(module, input, output):
    fmap_block.append(output.detach().cpu().numpy())

def backward_hook(module, grad_input, grad_output):
    grad_block.append(grad_output[0].detach().cpu().numpy())


# Load data and apply transformation
gloss_dict = np.load(dict_path, allow_pickle=True).item()
inputs_list = np.load(f"./preprocess/{dataset}/dev_info.npy", allow_pickle=True).item()
name = inputs_list[select_id]['fileid']
print(f'Generating CAM for {name}')
img_folder = os.path.join(prefix, "features/fullFrame-256x256px/" + inputs_list[select_id]['folder']) if 'phoenix' in dataset else os.path.join(prefix, inputs_list[select_id]['folder'])
img_list = sorted(glob.glob(img_folder))
img_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
label_list = []
for phase in inputs_list[select_id]['label'].split(" "):
    if phase == '':
        continue
    if phase in gloss_dict.keys():
        label_list.append(gloss_dict[phase][0])
transform = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])
vid, label = transform(img_list, label_list, None)
vid = vid.float() / 127.5 - 1
vid = vid.unsqueeze(0)

left_pad = 0
last_stride = 1
total_stride = 1
kernel_sizes = ['K5', "P2", 'K5', "P2"]
for layer_idx, ks in enumerate(kernel_sizes):
    if ks[0] == 'K':
        left_pad = left_pad * last_stride
        left_pad += int((int(ks[1])-1)/2)
    elif ks[0] == 'P':
        last_stride = int(ks[1])
        total_stride = total_stride * last_stride

max_len = vid.size(1)
video_length = torch.LongTensor([np.ceil(vid.size(1) / total_stride) * total_stride + 2*left_pad ])
right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
max_len = max_len + left_pad + right_pad
vid = torch.cat(
    (
        vid[0,0][None].expand(left_pad, -1, -1, -1),
        vid[0],
        vid[0,-1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
    )
    , dim=0).unsqueeze(0)


device = torch.device('cpu')
model = SLRModel(
    num_classes=len(gloss_dict) + 1,
    c2d_type='resnet34',
    conv_type=2,
    use_bn=1,
    gloss_dict=gloss_dict,
    loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0}
)


state_dict = torch.load(model_weights, map_location=device)['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.train()


# target_layer = model.conv2d.shift3.st_conv[4] if 'phoenix' in dataset else model.conv2d.shift3.st_conv[3]
target_layer = model.conv2d.stf1  if 'phoenix' in dataset else model.conv2d.stf1
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_backward_hook(backward_hook)


vid = vid.to(device)
vid_lgt = video_length.to(device)
label = torch.LongTensor(label).unsqueeze(0).to(device)
label_lgt = torch.LongTensor([len(label_list)]).to(device)

ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)


loss = model.criterion_calculation(ret_dict, label, label_lgt)
model.zero_grad()
loss.backward()


fmap = fmap_block[0]
grads_val = grad_block[0]


forward_handle.remove()
backward_handle.remove()


def cam_show_img(img, feature_map, grads, out_dir):
    N, C, T, H, W = feature_map.shape
    cam = np.zeros((T, H, W), dtype=np.float32)

    grads = grads.reshape(C, T, -1)
    weights = np.mean(grads, axis=-1)

    for i in range(C):
        for j in range(T):
            cam[j] += weights[i, j] * feature_map[0, i, j]

    cam = np.maximum(cam, 0)
    os.makedirs(out_dir, exist_ok=True)

    for t in range(T):

        cam_frame = cam[t] - np.min(cam[t])
        cam_frame /= (cam_frame.max() + 1e-8)


        cam_frame = cv2.resize(cam_frame, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_frame), cv2.COLORMAP_JET)


        img_frame = (img[0, t].permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255
        superimposed = cv2.addWeighted(img_frame.astype(np.uint8), 0.5, heatmap, 0.5, 0)

        cv2.imwrite(os.path.join(out_dir, f"frame_{t:03d}.jpg"), superimposed)

cam_show_img(vid, fmap, grads_val, './CAM')


