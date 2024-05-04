# ZigZag: Universal Sampling-free Uncertainty Estimation Through Two-Step Inference

![Project Page](./src/teaser.gif)

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2211.11435-blue?logo=arxiv&color=red)](https://arxiv.org/abs/2211.11435)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&color=blue)](https://www.python.org/downloads/release/python-31014/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.2.1-blue?logo=pytorch&color=blue)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/cvlab-epfl/zigzag/blob/main/LICENSE)

### [Project Page](https://www.norange.io/projects/zigzag/) | [OpenReview](https://openreview.net/forum?id=QSvb6jBXML) | [TMLR Paper](https://openreview.net/pdf?id=QSvb6jBXML)

> **For a quick tryout of the ZigZag method, check the Colab notebooks below!**

## Abstract

Whereas the ability of deep networks to produce useful predictions on many kinds of data has been amply demonstrated, estimating the reliability of these predictions remains challenging. Sampling approaches such as MC-Dropout and Deep Ensembles have emerged as the most popular ones for this purpose. Unfortunately, they require many forward passes at inference time, which slows them down. Sampling-free approaches can be faster but often suffer from other drawbacks, such as lower reliability of uncertainty estimates, difficulty of use, and limited applicability to different types of tasks and data.

In this work, we introduce a sampling-free approach that is generic and easy to deploy, while producing reliable uncertainty estimates on par with state-of-the-art methods at a significantly lower computational cost. It is predicated on training the network to produce the same output with and without additional information about it. At inference time, when no prior information is given, we use the network's own prediction as the additional information. We then take the distance between the predictions with and without prior information as our uncertainty measure.

## TL;DR

![Project Page](./src/arch.png)

**ZigZaging:** At inference time, we make two forward passes. First, we use $[\mathbf{x}, \textbf{0}]$ as input to produce a prediction $\mathbf{y}\_{0}$. Second, we feed $[\mathbf{x}, \mathbf{y}\_{0}]$ to the network and generate $\mathbf{y}\_{1}$. We take $\| \mathbf{y}\_{0} - \mathbf{y}\_{1} \|$ to be our uncertainty estimate. In essence, the second pass performs a reconstruction in much the same way an auto-encoder does and a high reconstruction error correlates with uncertainty.

**Motivation:** The second pass reconstructs the second input, expecting lower error for in-distribution data and higher for out-of-distribution, enabling uncertainty estimation. When given a correct label with input , the network, trained to minimize the difference between outputs, indicates in-distribution data. If is incorrect, this out-of-distribution sample prompts an unpredictable response, which we use to gauge uncertainty. This mechanism addresses both epistemic uncertainty when is OOD and aleatoric uncertainty when is errornous. 

## Experiments

### 1D Regression 

**Uncertainty Estimation for Regression:** The task is to regress $y$-axis values for $x$-axis data points drawn from the range $x \in [-1, 3]$ using a third-degree polynomial with added Gaussian noise. The method displays higher uncertainty or reconstruction error for out-of-distribution inputs and lower for in-distribution samples.

[![Open ZigZag in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvlab-epfl/zigzag/blob/main/exps/notebooks/toy_regression.ipynb)

### UCI Datasets Regression

TBD

[![Open ZigZag in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvlab-epfl/zigzag/blob/main/exps/notebooks/uci_regression.ipynb)

### MNIST Classification

**MNIST vs FashionMNIST:** We train the networks on MNIST and compute the accuracy and calibration metrics (rAULC). We then use the uncertainty measure they produce to classify images from the test sets of MNIST and FashionMNIST as being within the MNIST distribution or not to compute the OOD metrics, ROC- and PR-AUCs. We use a standard architecture with several convolution and pooling layers, followed by fully connected layers with LeakyReLU activations.

[![Open ZigZag in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvlab-epfl/zigzag/blob/main/exps/notebooks/mnist_classification.ipynb)

### CIFAR Classification

TBD

[![Open ZigZag in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

### Airfoils Regression

TBD

[![Open ZigZag in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

## Citation

If you find this code useful, please consider citing our paper:

> Durasov, Nikita, et al. "ZigZag: Universal Sampling-free Uncertainty Estimation Through Two-Step Inference." TMLR 20224.

```bibtex
@article{durasov2024zigzag,
    title = {ZigZag: Universal Sampling-free Uncertainty Estimation Through Two-Step Inference},
    author = {Nikita Durasov and Nik Dorndorf and Hieu Le and Pascal Fua},
    journal = {Transactions on Machine Learning Research},
    issn = {2835-8856},
    year = {2024}
}
``` 
