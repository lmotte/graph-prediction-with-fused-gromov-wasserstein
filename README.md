# Graph prediction with fused Gromov-Wasserstein

This repository contains a python implementation of the the supervised graph prediction method proposed in [] using PyTorch library and POT library (Python Optimal Transport).

## Method description

[] introduce a framework to solve supervised labeled graph prediction problems by leveraging optimal transport tools.

Let's explain how it works.

**Fused Gromow-Wasserstein (FGW) distance.**

Definition + a picture (titouan + credit to him email).

**FGW as a loss.**

**FGW barycentric graph prediction model.**

**Two training methods.** [] proposed 1) a non-parametric and 2) a neural network approaches to train the proposed model. We advice to use the first method on small datasets, and the second method on big datasets.


## How to test the method on the synthetic graph prediction problem

![TEST](illustrations/illu_deep2.pdf)

## How to test the method on any graph prediction problem

dependencies

step-by-step

