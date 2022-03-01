<img src='source/UFO.png'>

--------------------------------------------------------------------------------

# A Unified Transformer Framework for Group-based Segmentation: Co-Segmentation, Co-Saliency Detection and Video Salient Object Detection

**[[arxiv](https://scholar.google.com/citations?user=O00rbxoAAAAJ&hl=zh-CN)]** **[[supp](https://scholar.google.com/citations?user=O00rbxoAAAAJ&hl=zh-CN)]**

UFO is a simple and Unified framework for addressing Co-Object Segmentation tasks: Co-Segmentation, Co-Saliency Detection and Video Salient Object Detection

Humans tend to mine objects by learning from a group of images or a several frames of video since we live in a dynamic world. In computer vision area, many researches focus on co-segmentation (CoS), co-saliency detection (CoSD) and video salient object detection (VSOD) to discover the co-occurrent objects. However, previous approaches design different networks on these tasks separately, which lower the upper bound on the ease of use of deep learning frameworks. In this paper, we introduce a unified framework to tackle these issues, term as **UFO** (**U**nified **F**ramework for Co-**O**bject Segmentation). All tasks share the same framework.

## Framework

<img src='source/framework.png'>

## Result

- **Co-Segmentation (CoS) on [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [iCoseg](http://chenlab.ece.cornell.edu/projects/touch-coseg/), [Internet](http://people.csail.mit.edu/mrub/ObjectDiscovery/) and [MSRC](https://link.springer.com/chapter/10.1007/11744023_1)  [[Pre-computed Results](https://drive.google.com/drive/folders/1aLNYQDeG6ibbxsfI686TJKRbjj6nDF0U?usp=sharing)]** 

<img src='source/result1.png'>

- **Co-Saliency Detection(CoSD) on [CoCA](http://zhaozhang.net/coca.html)，[CoSOD3k](http://dpfan.net/CoSOD3K/) and [CoSal2015](https://ieeexplore.ieee.org/abstract/document/7298918)  [[Pre-computed Results](https://drive.google.com/drive/folders/1QCr0zCCIsBC7JEHBS6A1O3V2JIpEAHyr?usp=sharing)]**

<img src='source/result2.png'>

- **Video Salient Object Detection (VSOD) on [DAVIS16 val set](https://davischallenge.org/davis2016/code.html)** 

<img src="source/drift-straight.gif" width="45%"/> <img src="source/bmx-trees.gif" width="45%"/>

- **[Optional] Single Object Tracking (SOT) on [GOT-10k val set](http://got-10k.aitestunion.com/downloads)** 

<img src="source/bear_480p.gif" width="45%"/><img src="source/rabbit_480p.gif" width="45%"/>


## Usage

--TODO--

