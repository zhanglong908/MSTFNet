# MSTFNet
![image](https://raw.githubusercontent.com/zhanglong908/MSTFNet/main/framework.png?t=20250917)

This repo holds codes of the paper: MSTFNet: Multi-Scale Spatio-Temporal Fusion Network with Hierarchical Feature Alignment for Continuous Sign Language Recognition

This repo is based on [CorrNet](https://github.com/hulianyuyy/CorrNet_CSLR). Many thanks for their great work!


## Prerequisites

- This project is implemented in Pytorch (better >=1.13 to be compatible with ctcdecode or these may exist errors). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode. (ctcdecode is only supported on the Linux platform.)

- [Optional] sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite: 
  `mkdir ./software`
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`

   You may use the python version evaluation tool for convenience (by setting 'evaluate_tool' as 'python' in line 16 of ./configs/baseline.yaml), but sclite can provide more detailed statistics.

- You can install other required modules by conducting 
   `pip install -r requirements.txt`

## Implementation
We found that on the PHOENIX2014 dataset, the best performance is achieved when the HFAM module shares a common HA across all layers, whereas on PHOENIX2014-T and CSL-Daily, assigning layer-specific HAs yields the optimal results.
We attribute this discrepancy to the fact that PHOENIX2014 contains shorter, more repetitive sentence patterns whose temporal dynamics can be 
adequately captured by a single, shared attention head; in contrast, the longer and linguistically richer utterances in PHOENIX2014-T and CSL-Daily 
benefit from layer-wise specialized attention heads that progressively refine distinct aspects of spatio-temporal dependencies.

## Data Preparation
We carefully selected three authoritative and widely-adopted public datasets that together cover the main application scenarios, vocabulary scales and linguistic complexities of continuous sign language recognition.
Together these datasets span weather news, daily dialogue and multi-modal scenarios, providing complementary challenges such as large vocabulary, long-range temporal dependencies, signer variation and visual diversity; consequently they constitute a rigorous test-bed for verifying the effectiveness and generalization ability of MSTFNet.
You can choose any one of the following datasets to reproduce the reported results or further verify the effectiveness of MSTFNet.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-T.py --process-image --multiprocessing
   ```

If you get an error like ```IndexError: list index out of range``` on the PHOENIX2014-T dataset, you may refer to [this issue](https://github.com/hulianyuyy/CorrNet/issues/10#issuecomment-1660363025) to tackle the problem.



### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL-Daily`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-CSL-Daily.py --process-image --multiprocessing
   ``` 

## Inference

### PHOENIX2014 dataset

| Backbone | Dev WER | Test WER | Pretrained model                                             |
|----------|---------|----------| --- |
| ResNet34 | 17.1%   | 17.6%    | [[Baidu]]( https://pan.baidu.com/s/1ZmobWAegBtuoFJNFqO85Yg) (passwd: 9kfw)<br /> |



### PHOENIX2014-T dataset

| Backbone | Dev WER | Test WER | Pretrained model                                             |
|----------|---------|----------| --- |
| ResNet34 | 16.8%   | 17.9%    | [[Baidu]](https://pan.baidu.com/s/1YKSoMxXKShEcGQEUGgiC3Q) (passwd: i5y3)<br />|

### CSL-Daily dataset

| Backbone | Dev WER | Test WER | Pretrained model                                            |
|----------|---------|----------| --- |
| ResNet34 | 24.8%   | 24.0%    | [[Baidu]]( https://pan.baidu.com/s/1VODfeKYzzdcyi8dzwRpTJw) (passwd: hpcn)<br />|

We wrongly delete the original checkpoint and retrain the model with similar accuracy (Dev: 24.8%, Test: 24.0%)
​	

To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：   
`python main.py --config ./config/baseline.yaml --device your_device --load-weights path_to_weight.pt --phase test`

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --config ./config/baseline.yaml --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.


### Visualizations
For Grad-CAM visualization, you can  run ```python generate_cam.py``` with your own hyperparameters.
![image](https://raw.githubusercontent.com/zhanglong908/MSTFNet/main/heatmap.png)
### Test with one video input
Except performing inference on datasets, we provide a `test_one_video.py` to perform inference with only one video input. An example command is 

`python test_one_video.py --model_path /path_to_pretrained_weights --video_path /path_to_your_video --device your_device`

The `video_path` can be the path to a video file or a dir contains extracted images from a video.

Acceptable paramters:
- `model_path`, the path to pretrained weights.
- `video_path`, the path to a video file or a dir contains extracted images from a video.
- `device`, which device to run inference, default=0.
- `language`, the target sign language, default='phoenix', choices=['phoenix', 'csl-daily'].
- `max_frames_num`, the max input frames sampled from an input video, default=360.
