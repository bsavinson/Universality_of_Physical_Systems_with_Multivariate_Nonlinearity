# Universality of Physical Systems with Multivariate Nonlinearity

This repository contains the code and saved outputs needed to reproduce the MNIST and FashionMNIST twin plots for the paper "Universality of Physical Systems with Multivariate Nonlinearity".

It includes the free-space physical neural network model, the training script, pretrained runs, and the plotting script used to regenerate the paper figures.

## Repository Layout

- `free_space.py` - Free-space physical neural network model used by the paper plots.
- `train.py` - Training script for the supported MNIST/FashionMNIST configuration.
- `plot_paper_twinplots.py` - Regenerates the twin plots from saved accuracy data and pretrained weights.
- `data/` - Precomputed `.npz` accuracy-vs-replication data used by the plotting script.
- `runs/` - Model checkpoints used for the confusion matrices.
- `figs/` - Generated figure outputs.

## Supported Configuration

The code supports the configuration used for the paper figures:

- MNIST and FashionMNIST inputs resized to `14 x 14`
- phase encoding
- one optical layer
- S matrix
- multi-lens 10-class readout
- SGD training with autograd
- optional training of the S matrix via `--train_S` / `--no-train_S`

## Setup

Use a Python environment with PyTorch, torchvision, NumPy, Matplotlib, Seaborn, and scikit-learn installed.

For example:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

The code uses CUDA when available and otherwise falls back to CPU.

## Reproducing the Figures

Run:

```bash
python plot_paper_twinplots.py
```

This writes:

- `figs/twinplot_accuracy_confusion_digitmnist_paper.png`
- `figs/twinplot_accuracy_confusion_fashionmnist_paper.png`

The accuracy curves are loaded from `data/*.npz`. The confusion matrices are recomputed from the saved model checkpoints in `runs/`, so this step can be slow on CPU. The script prints how many test samples have been evaluated while it runs.

## Training

To train a new DigitMNIST model:

```bash
python train.py --replications_per_dim 1 --train_S
```

To train a new FashionMNIST model:

```bash
python train.py --fashion_mnist --replications_per_dim 1 --train_S
```

Useful options:

- `--replications_per_dim` controls the input replication factor per spatial dimension.
- `--train_S` trains the S matrix.
- `--no-train_S` keeps the randomly initialized S matrix fixed.
- `--epochs`, `--batch_size`, `--learning_rate`, and `--momentum` control the optimizer run.

Datasets are downloaded by torchvision into `mnist_data/` if they are not already present.

## Notes for Public Use

The saved checkpoints in `runs/` are loaded as PyTorch state dictionaries. Only use checkpoints from sources you trust.
