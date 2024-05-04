# ZigZag: Universal Sampling-free Uncertainty Estimation Through Two-Step Inference

![Project Page](./src/teaser.gif)

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2211.11435-blue?logo=arxiv&color=red)](https://arxiv.org/abs/2211.11435)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&color=blue)](https://www.python.org/downloads/release/python-31014/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.2.1-blue?logo=pytorch&color=blue)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/cvlab-epfl/zigzag/blob/main/LICENSE)

### [Project Page](https://www.norange.io/projects/zigzag/) | [OpenReview](https://openreview.net/forum?id=QSvb6jBXML) | [Paper](https://arxiv.org/abs/2211.11435)

## Abstract

Whereas the ability of deep networks to produce useful predictions on many kinds of data has been amply demonstrated, estimating the reliability of these predictions remains challenging. Sampling approaches such as MC-Dropout and Deep Ensembles have emerged as the most popular ones for this purpose. Unfortunately, they require many forward passes at inference time, which slows them down. Sampling-free approaches can be faster but often suffer from other drawbacks, such as lower reliability of uncertainty estimates, difficulty of use, and limited applicability to different types of tasks and data.

In this work, we introduce a sampling-free approach that is generic and easy to deploy, while producing reliable uncertainty estimates on par with state-of-the-art methods at a significantly lower computational cost. It is predicated on training the network to produce the same output with and without additional information about it. At inference time, when no prior information is given, we use the network's own prediction as the additional information. We then take the distance between the predictions with and without prior information as our uncertainty measure.

## TL;DR

![Project Page](./src/arch.png)

## Experiments

### 1D Regression

[![Open Masksembles in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvlab-epfl/zigzag/blob/main/exps/notebooks/toy_regression.ipynb)

### UCI Datasets Regression

[![Open Masksembles in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvlab-epfl/zigzag/blob/main/exps/notebooks/uci_regression.ipynb)

### MNIST Classification

[![Open Masksembles in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvlab-epfl/zigzag/blob/main/exps/notebooks/mnist_classification.ipynb)