# CORA Benchmark for VNNCOMP 2024

We propose a benchmark for the [VNNCOMP 2024](https://sites.google.com/view/vnn2024) that focuses on verification time.
The benchmark consists of one ReLU-neural network (7x250 + ReLU), which was trained on three datasets, i.e., MNIST, SVHN, and CIFAR10, using three different training methods, i.e., standard (point), interval-bound propagation, and set-based. The neural networks are taken from the first evaluation run of [1]; please refer to [1] for the training details.

## References
[1] Koller, Lukas, Tobias Ladner, and Matthias Althoff. "End-To-End Set-Based Training for Neural Network Verification." [arXiv preprint arXiv:2401.14961](https://arxiv.org/abs/2401.14961) (2024).
