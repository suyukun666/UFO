<img src='source/UFO.png'>

--------------------------------------------------------------------------------

# A Unified Transformer Framework for Group-based Segmentation: Co-Segmentation, Co-Saliency Detection and Video Salient Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-transformer-framework-for-group/co-salient-object-detection-on-cosal2015)](https://paperswithcode.com/sota/co-salient-object-detection-on-cosal2015?p=a-unified-transformer-framework-for-group)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-transformer-framework-for-group/co-salient-object-detection-on-cosod3k)](https://paperswithcode.com/sota/co-salient-object-detection-on-cosod3k?p=a-unified-transformer-framework-for-group)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-transformer-framework-for-group/co-salient-object-detection-on-coca)](https://paperswithcode.com/sota/co-salient-object-detection-on-coca?p=a-unified-transformer-framework-for-group)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-transformer-framework-for-group/video-salient-object-detection-on-segtrack-v2)](https://paperswithcode.com/sota/video-salient-object-detection-on-segtrack-v2?p=a-unified-transformer-framework-for-group)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-transformer-framework-for-group/video-salient-object-detection-on-visal)](https://paperswithcode.com/sota/video-salient-object-detection-on-visal?p=a-unified-transformer-framework-for-group)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-transformer-framework-for-group/video-salient-object-detection-on-fbms-59)](https://paperswithcode.com/sota/video-salient-object-detection-on-fbms-59?p=a-unified-transformer-framework-for-group)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-transformer-framework-for-group/video-salient-object-detection-on-davis-2016)](https://paperswithcode.com/sota/video-salient-object-detection-on-davis-2016?p=a-unified-transformer-framework-for-group)

**[[arxiv](https://arxiv.org/abs/2203.04708v1)]** 

UFO is a simple and Unified framework for addressing Co-Object Segmentation tasks: Co-Segmentation, Co-Saliency Detection and Video Salient Object Detection. Humans tend to mine objects by learning from a group of images or a several frames of video since we live in a dynamic world. In computer vision area, many researches focus on co-segmentation (CoS), co-saliency detection (CoSD) and video salient object detection (VSOD) to discover the co-occurrent objects. However, previous approaches design different networks on these tasks separately, which lower the upper bound on the ease of use of deep learning frameworks. In this paper, we introduce a unified framework to tackle these issues, term as <b>UFO</b> (<b>U</b>nified <b>F</b>ramework for Co-<b>O</b>bject Segmentation). All tasks share the same framework.        

## Task & Framework

<img src="source/fig1.gif" width="50%"/><img src='source/framework.png' width="50%">

## Usage

### Requirement

```python
torch >= 1.7.0
torchvision >= 0.7.0
python3
```

### Training

Training on group-based images. We use [COCO2017 train set](https://cocodataset.org/#home) with the provided [group split dict.npy](https://drive.google.com/file/d/1l-KY8JtUu1pfQ4Xd3s0JNrkLM69oT_Ud/view?usp=sharing).

```bash
python main.py image
```

Training on video. We load the weight pre-trained on the static image dataset, and use DAVIS and FBMS to train our network.

```bash
python main.py video --wo flow
```

Training on video (w/ fow). The same as above, we use DAVIS_flow and FBMS_flow to train our network.

```bash
python main.py video --w flow
```

### Inference

Generate the image results [[checkpoint](https://drive.google.com/file/d/1ZFJwxBFTekAAxGuDMoafP4slTS_dBe3O/view?usp=sharing)]

```bash
python eval.py image
```

Generate the video results [[checkpoint](https://drive.google.com/file/d/1eIAoCy-sV_9ueC9-KmQKDyc8nex2yWxL/view?usp=sharing)]

```bash
python eval.py video --wo flow
```

Generate the video results with optical flow [[checkpoint](https://drive.google.com/file/d/1NtX86od0jlukYlF2EIKsFYxZdjBW2pL6/view?usp=sharing)]

```bash
python eval.py video --w flow
```

### Evaluation

- Pre-Computed Results: Please download the prediction results of our framework form the Results section.
- Evaluation Toolbox: We use the standard evaluation toolbox from [COCA benchmark](http://zhaozhang.net/coca.html).

## Result

- **Co-Segmentation (CoS) on [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [iCoseg](http://chenlab.ece.cornell.edu/projects/touch-coseg/), [Internet](http://people.csail.mit.edu/mrub/ObjectDiscovery/) and [MSRC](https://link.springer.com/chapter/10.1007/11744023_1)  [[Pre-computed Results](https://drive.google.com/drive/folders/1aLNYQDeG6ibbxsfI686TJKRbjj6nDF0U?usp=sharing)]** 

<img src='source/result1.png'>

- **Co-Saliency Detection(CoSD) on [CoCA](http://zhaozhang.net/coca.html)，[CoSOD3k](http://dpfan.net/CoSOD3K/) and [CoSal2015](https://ieeexplore.ieee.org/abstract/document/7298918)  [[Pre-computed Results](https://drive.google.com/drive/folders/1QCr0zCCIsBC7JEHBS6A1O3V2JIpEAHyr?usp=sharing)]**

<img src='source/result2.png'>

- **Video Salient Object Detection (VSOD) on [DAVIS16 val set](https://davischallenge.org/davis2016/code.html)  [[Pre-computed Results](https://drive.google.com/drive/folders/1iv6Rrdn3r2S5g5BSdViXT-vTX1pJUlBq?usp=sharing)]** 

<img src="source/drift-straight.gif" width="45%"/> <img src="source/bmx-trees.gif" width="45%"/>

- **[Optional] Single Object Tracking (SOT) on [GOT-10k val set](http://got-10k.aitestunion.com/downloads)** 

<img src="source/bear_480p.gif" width="45%"/><img src="source/rabbit_480p.gif" width="45%"/>

## Demo

https://user-images.githubusercontent.com/50760123/156528285-59b0a056-fb07-4c1e-8e66-cae31dc0e789.mp4

https://user-images.githubusercontent.com/50760123/156924040-c329075f-1d50-41cd-a869-885b2f33d873.mp4

## Acknowledgement

Our project references the codes in the following repos.

- [SSNM](https://github.com/cj4L/SSNM-Coseg)
- [VIT](https://github.com/google-research/vision_transformer)
