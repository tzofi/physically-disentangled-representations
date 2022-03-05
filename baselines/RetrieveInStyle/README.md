# Retrieve in Style and StyleGAN2 Baselines

The following code was adapted from [this repo](https://github.com/mchong6/RetrieveInStyle). The authors thank them for their work.

Code for generating features for clustering is in extract.py (RAF-DB), extract\_multimedia.py (BBT, Buffy), and extract\_celeba.py (CelebA).

## Retrieve in Style: Unsupervised Facial Feature Transfer and Retrieval PyTorch
![](teaser.gif)
![](ris_teaser.png)

This is the PyTorch implementation of [Retrieve in Style: Unsupervised Facial Feature Transfer and Retrieval](https://arxiv.org/abs/2107.06256). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mchong6/RetrieveInStyle/blob/main/RIS_colab.ipynb)


>**Abstract:**<br>
>We present Retrieve in Style (RIS), an unsupervised framework for fine-grained facial feature transfer and retrieval on real images Recent work shows that it is possible to learn a catalog that allows local semantic transfers of facial features on generated images by capitalizing on the disentanglement property of the StyleGAN latent space. RIS improves existing art on: 
>1) feature disentanglement and allows for challenging transfers (\ie, hair and pose) that were not shown possible in SoTA methods.
>2) eliminating the needs for per-image hyperparameter tuning, and for computing a catalog over a large batch of images.
>3) enabling face retrieval using the proposed facial features (\eg, eyes), and to our best knowledge, is the first work to retrieve face images at the fine-grained level.
>4) robustness and natural application to real images. 
>Our qualitative and quantitative analyses show RIS achieves both high-fidelity feature transfers and accurate fine-grained retrievals on real images. 
>We discuss the responsible application of RIS.

## Dependency
Our codebase is based off [stylegan2 by rosalinity](https://github.com/rosinality/stylegan2-pytorch). 
```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install tqdm gdown scikit-learn scipy lpips dlib opencv-python
```

## How to use
Everything to get started is in the [colab notebook](https://colab.research.google.com/github/mchong6/RetrieveInStyle/blob/main/RIS_colab.ipynb).

## Citation
If you use this code or ideas from our paper, please cite our paper:
```
@InProceedings{Chong_2021_ICCV,
    author    = {Chong, Min Jin and Chu, Wen-Sheng and Kumar, Abhishek and Forsyth, David},
    title     = {Retrieve in Style: Unsupervised Facial Feature Transfer and Retrieval},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {3887-3896}
}
```

## Acknowledgments
This code borrows from [StyleGAN2 by rosalinity](https://github.com/rosinality/stylegan2-pytorch), [Editing in Style](https://github.com/IVRL/GANLocalEditing), [StyleClip](https://github.com/orpatashnik/StyleCLIP), [PTI](https://github.com/danielroich/PTI). Encoder used is borrowed directly from [encoder4editing](https://github.com/omertov/encoder4editing).
