# Graph prediction with fused Gromov-Wasserstein

This repository contains a python implementation of the supervised learning method proposed in [Brogat-Motte et al., 2022](#references) using PyTorch library and POT library (Python Optimal Transport).

This method aims providing a general method for solving (labeled) graph prediction problems. It takes advantage of recent advances in computational optimal transport. In particular, it makes use of the FGW distance ([Vayer et al., 2020](#references)) which is a natural metric for graph comparison.
In particular, FGW distance allows to leverage a ground metric on the graphs's nodes. For example, depending on the task at hand, different distances between atoms can be used to defined the FGW distance over the molecular graphs space. All details about the method are provided in [Brogat-Motte et al., 2022](#references).

Two versions of the method are provided in this repository: a non-parametric approach and a neural network approach.

## Quick start code example

**Load data.**

```python
pip install -r requirements.txt
```


## References

- Brogat-Motte et al.
- Vayer blabli
