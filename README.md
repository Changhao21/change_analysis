# Overview

This repository brings together our work on remote sensing change detection,
change captioning, and general-purpose pixel-wise segmentation.

If you find our work useful, please consider citing the corresponding papers.
For questions, corrections, or omissions, please contact us at
[changhao21@mails.ucas.ac.cn](mailto:changhao21@mails.ucas.ac.cn) or
[changhao281@nudt.edu.cn](mailto:changhao281@nudt.edu.cn).

We gratefully acknowledge the authors of the related datasets and codebases
listed below.

## 1. Binary Change Detection

### Dataset

We use the dataset provided by [DDPM-CD](https://github.com/wgcban/ddpm-cd).

### Training Code

Our training pipeline is based on [ChangeFormer](https://github.com/wgcban/ChangeFormer).

### Our Implementation

[`MDIPNet.py`](./MDIPNet.py) implements our method:

> Remote Sensing Change Detection With Bitemporal and Differential Feature
> Interactive Perception, *IEEE Transactions on Image Processing*, 2024.

```bibtex
@ARTICLE{10599227,
  author={Chang, Hao and Wang, Peijin and Diao, Wenhui and Xu, Guangluan and Sun, Xian},
  journal={IEEE Transactions on Image Processing},
  title={Remote Sensing Change Detection With Bitemporal and Differential Feature Interactive Perception},
  year={2024},
  volume={33},
  number={},
  pages={4543-4555},
  keywords={Transformers; Semantics; Attention mechanisms; Correlation; Feature extraction; Costs; Computational modeling; Remote sensing; Change detection; Change of interest; Semantic correlation},
  doi={10.1109/TIP.2024.3424335}
}
```

## 2. Semantic Change Detection

### Dataset

We processed the dataset ourselves. Please refer to our paper for details.

### Training Code

Our training pipeline is based on [Bi-SRNet](https://github.com/ggsDing/Bi-SRNet).

### Our Implementation

[`JFRNet.py`](./JFRNet.py) implements our method:

> A Triple-Branch Hybrid Attention Network With Bitemporal Feature Joint
> Refinement for Remote-Sensing Image Semantic Change Detection,
> *IEEE Transactions on Geoscience and Remote Sensing*, 2024.

```bibtex
@ARTICLE{10466771,
  author={Chang, Hao and Wang, Peijin and Diao, Wenhui and Xu, Guangluan and Sun, Xian},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={A Triple-Branch Hybrid Attention Network With Bitemporal Feature Joint Refinement for Remote-Sensing Image Semantic Change Detection},
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Semantics; Decoding; Task analysis; Correlation; Feature extraction; Data mining; Convolution; Attention mechanism; Branch interaction; Convolutional neural network (CNN); Semantic change detection (SCD)},
  doi={10.1109/TGRS.2024.3376384}
}
```

## 3. Change Captioning

Coming soon.

## 4. Pixel-Wise Semantic Segmentation

[`DynaPM.py`](./DynaPM.py) implements our method:

> DynaPM: Dynamic Pixel Mining With Spatial Mask Guidance for Semantic
> Segmentation Optimization, *IEEE Transactions on Geoscience and Remote
> Sensing*, 2026.

```bibtex
@ARTICLE{11770360,
  author={Chang, Hao and Liu, Ting and Li, Boyang and Li, Weixing and Chen, Huaiyu and Fan, JianPeng and Liu, Zijing and Cheng, Gong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={DynaPM: Dynamic Pixel Mining With Spatial Mask Guidance for Semantic Segmentation Optimization},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Pixel; Modeling; Optimization; Learning (artificial intelligence); Labeling; Semantic segmentation; Training; Conferences; Remote sensing; Image segmentation; Pixel imbalance; Sample mining; Mask guidance},
  doi={10.1109/TGRS.2026.3749585}
}
```
