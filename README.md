# Physically Disentangled Representations

This repository contains the code used for Physically Disentangled Representations (in submission).

## Overview

Our proposed method builds off existing inverse rendering techniques to learn physically disentangled representations that can be used across many downstream tasks. Our repository contains the code used to train the inverse renderer used in our method, both with or without our proposed Leave-One-Out, Cycle Contrastive loss (LOOCC). We then provide the code used for evaluating our model on downstream clustering, linear classification, and segmentation tasks. Lastly, the repo contains the code used to benchmark other methods against our own, so that all results in our paper can easily be reproduced.

## Model Training

Our inverse rendering code builds off of [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://github.com/elliottwu/unsup3d). We would like to thank the authors of that work for their code. We provide the config files used to train the faces and cars models, which were trained on UTK Faces and ShapeNet cars, respectively, in the experiments directory. To train the model:

```
python run.py --config experiments/train_utkfaces.yaml --gpu 0 --num_workers 4
```

The `cycle` parameter in the config can be set to true to enable LOOCC, and otherwise should be set to false.

## Model Evaluation

### Clustering

We evaluate our model on hierarchical agglomerative clustering to measure the utility of raw features. Please refer to cluster\_eval.py for implementation. Given a set of features and corresponding labels, cluster\_eval.py will report clustering accuracy, F1, and normalized mutual information (NMI).

```
python cluster_eval.py data.npy
```

where data.npy contains the features and labels in the "features" and "labels" keys.

This file can be found in the clustering directory.

### Linear Classification

We follow the standard protocol to evaluate our model on linear classification by attaching an MLP projection head onto our pre-trained frozen or trainable model. To train the MLP, we use run\_trainer.py in the classification directory:

```
python run_trainer.py
```

### Segmentation

We leverage the implementation of U-net from [here]() for face segmentation.

To run supervised baseline:

```
python run.py
```

To use pretrained encoder:

```
python run.py path_to_pretrained_model/ckpt.pt
``` 

## Baselines

Please see the baselines folder to see the code used for generating baselines for this paper, including VQ-VAE, StyleGAN2, and Retrieve in Style.
