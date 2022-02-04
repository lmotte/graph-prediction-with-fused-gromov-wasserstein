# Graph prediction with fused Gromov-Wasserstein

This repository contains a python implementation of the the supervised graph prediction method proposed in [] using PyTorch library and POT library (Python Optimal Transport).

- [Method description](#method-description)
- [How to test the method on the synthetic graph prediction problem](#how-to-test-the-method-on-the-synthetic-graph-prediction-problem)
- [How to test the method on any graph prediction problem](#how-to-test-the-method-on-any-graph-prediction-problem)

 
## Method description

[Brogat-Motte et al., 2021](#references) introduce a framework to solve supervised labeled graph prediction problems by leveraging optimal transport tools.

We provide here a short description of its functioning.

**Fused Gromow-Wasserstein (FGW) distance.** The FGW distance has been proposed recently as an extension of Gromov-Wasserstein distance to measure the imilarity between attributed
graphs [Vayer et al., 2020](#references).

Definition

Example of molecular graphs space.

a picture (titouan + credit to him email).

**FGW as a loss.**

**FGW barycentric graph prediction model.**

**Two training methods.** [] proposed 1) a non-parametric and 2) a neural network approaches to train the proposed model. We advice to use the first method on small datasets, and the second method on big datasets.


## How to test the method on the synthetic graph prediction problem

![Model](illustrations/illu_deep2.pdf)

xfun::embed_file("illustrations/illu_deep2.pdf")


## How to test the method on any graph prediction problem

dependencies

step-by-step

## References

- Brogat-Motte et al.
- Vayer blabli
