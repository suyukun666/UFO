<img src='source/UFO.png'>

--------------------------------------------------------------------------------

# Video Inpainting

We extend our <b>UFO</b> (<b>U</b>nified <b>F</b>ramework for Co-<b>O</b>bject Segmentation) framework to more application using the same network architecture like video inpainting.

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
python train.py
```

### Inference

Generate the image results [[checkpoint](https://drive.google.com/file/d/1ZFJwxBFTekAAxGuDMoafP4slTS_dBe3O/view?usp=sharing)]

```bash
python test.py 
```

## Result

- **Co-Segmentation (CoS) on [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [iCoseg](http://chenlab.ece.cornell.edu/projects/touch-coseg/), [Internet](http://people.csail.mit.edu/mrub/ObjectDiscovery/) and [MSRC](https://link.springer.com/chapter/10.1007/11744023_1)  [[Pre-computed Results](https://drive.google.com/drive/folders/1aLNYQDeG6ibbxsfI686TJKRbjj6nDF0U?usp=sharing)]** 

<img src='source/result1.png'>



## Demo

```bash
python demo.py --data_path=./demo_mp4/video/kobe.mp4 --output_dir=./demo_mp4/result
```

https://user-images.githubusercontent.com/50760123/156528285-59b0a056-fb07-4c1e-8e66-cae31dc0e789.mp4



## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```latex
@misc{2203.04708,
Author = {Yukun Su and Jingliang Deng and Ruizhou Sun and Guosheng Lin and Qingyao Wu},
Title = {A Unified Transformer Framework for Group-based Segmentation: Co-Segmentation, Co-Saliency Detection and Video Salient Object Detection},
Year = {2022},
Eprint = {arXiv:2203.04708},
}

```


## Acknowledgement

Our project references the codes in the following repos.

- [SSNM](https://github.com/cj4L/SSNM-Coseg)
- [VIT](https://github.com/google-research/vision_transformer)



