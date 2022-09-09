<img src='source/UFO.png'>

--------------------------------------------------------------------------------

# Video Inpainting

We extend our <b>UFO</b> (<b>U</b>nified <b>F</b>ramework for Co-<b>O</b>bject Segmentation) framework to more applications using the same network architecture like video inpainting.

## Usage

### Requirement

```python
torch >= 1.7.0
torchvision >= 0.7.0
python3
```

### Training

Training on [YouTube Video Object Segmentation dataset 2019 version](https://youtube-vos.org/dataset/). Users can also train on their own datasets by following the Youtube dataset format.

```bash
python train.py
```

### Inference

Generate the video inpainting results [[checkpoint](https://drive.google.com/file/d/1ZFJwxBFTekAAxGuDMoafP4slTS_dBe3O/view?usp=sharing)]

```bash
python test.py 
```

## Demo

```bash
python demo.py --data_path=./demo_mp4/video/kobe.mp4 --output_dir=./demo_mp4/result
```





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

- [FuseFormer](https://github.com/ruiliu-ai/FuseFormer)
- [VIT](https://github.com/google-research/vision_transformer)



