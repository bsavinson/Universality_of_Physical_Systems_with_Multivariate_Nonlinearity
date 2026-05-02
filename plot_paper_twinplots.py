import os
import re
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.ticker import ScalarFormatter
from free_space import FreeSpaceModel


def get_replications_per_dim_from_path(path):
    match = re.search(r"replications_per_dim(\d+)", path)
    if match:
        return int(match.group(1))
    raise ValueError("replications_per_dim not found in path")


def load_model(path, train_S=False, replications_per_dim=1, device=None):
    model_filename = path.replace(".pt", "_best_model.pt")
    model_path = os.path.join("runs", model_filename)

    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(0)

    model = FreeSpaceModel(
        model_n=14,
        train_S=train_S,
        replications_per_dim=replications_per_dim,
        amplitude_encoding=False,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def plot_accuracies(
    accuracies_random,
    replications_random,
    accuracies_trained,
    replications_trained,
    filename=None,
    colors=None,
    yticks=None,
    ax=None,
):
    if colors is None:
        colors = ["red", "darkred"]

    input_replications_random = np.array(replications_random) ** 2
    accuracies_random = np.array(accuracies_random) * 100
    input_replications_trained = np.array(replications_trained) ** 2
    accuracies_trained = np.array(accuracies_trained) * 100

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(input_replications_random, accuracies_random, color=colors[1], s=80, label="Random S")
    ax.scatter(input_replications_trained, accuracies_trained, color=colors[0], s=80, label="Trained S")

    if len(input_replications_trained) > 1:
        log_x = np.log10(input_replications_trained)
        coeffs = np.polyfit(log_x, accuracies_trained, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(log_x.min(), log_x.max(), 100)
        ax.plot(10 ** x_fit, fit_line(x_fit), color=colors[0], linestyle="--", alpha=0.7)

    if len(input_replications_random) > 1:
        log_x = np.log10(input_replications_random)
        coeffs = np.polyfit(log_x, accuracies_random, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(log_x.min(), log_x.max(), 100)
        ax.plot(10 ** x_fit, fit_line(x_fit), color=colors[1], linestyle="--", alpha=0.7)

    ax.set_xscale("log")
    ax.set_xticks([1, 4, 9, 16])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticklabels(["1", "4", "9", "16"])

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_xlabel("Input replications", fontsize=16)
    ax.set_ylabel("Test accuracy", fontsize=16)
    ax.tick_params(axis="both", which="both", labelsize=16)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color("black")


def plot_confusion(model, test_loader, ax=None, cmap="Reds", class_names=None):
    if ax is None:
        _, ax = plt.subplots()

    all_preds = []
    all_targets = []
    total_samples = len(test_loader.dataset)
    evaluated_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(model.device if hasattr(model, "device") else next(model.parameters()).device)
            output = model(data.view(data.size(0), -1))
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.numpy())
            evaluated_samples += len(target)
            print(
                f"Evaluated {evaluated_samples}/{total_samples} samples",
                flush=True,
            )

    acc = accuracy_score(all_targets, all_preds)
    print(f"Test accuracy: {acc*100:.2f}%")

    cm = confusion_matrix(all_targets, all_preds, normalize="true")
    cm_percent = np.round(cm * 100).astype(int)

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt="d",
        cmap=cmap,
        vmin=0,
        vmax=100,
        square=True,
        annot_kws={"size": 18},
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted class", fontsize=16, labelpad=10)
    ax.set_ylabel("True class", fontsize=16, labelpad=10)

    if class_names is None:
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]

    ax.set_xticklabels([])
    ax.set_yticklabels(class_names, fontsize=16, rotation=0)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color("black")


def get_mnist_test_loader(dataset_cls):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((14, 14)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_loader = torch.utils.data.DataLoader(
        dataset_cls("./mnist_data", train=False, download=True, transform=transform),
        batch_size=128,
        shuffle=False,
    )
    return test_loader


def main():
    os.makedirs("./figs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}")

    digit_npz = "data/accuracy_replications_digitmnist_paper.npz"
    fashion_npz = "data/accuracy_replications_fashionmnist_paper.npz"
    print(f"[main] loading npz: {digit_npz}")
    print(f"[main] loading npz: {fashion_npz}")

    digit_confusion_run = "DigitMNIST_SGD_PhaseEncoding_model_n14_replications_per_dim3_lr0.001_momentum0.0_batchsize32_trainSTrue.pt"
    fashion_confusion_run = "FashionMNIST_SGD_PhaseEncoding_model_n14_replications_per_dim4_lr0.001_momentum0.0_batchsize32_trainSTrue.pt"

    digit_data = np.load(digit_npz)
    fashion_data = np.load(fashion_npz)
    print("[main] loaded npz files")

    print("[main] loading MNIST test loader")
    digit_test_loader = get_mnist_test_loader(torchvision.datasets.MNIST)
    print("[main] loading FashionMNIST test loader")
    fashion_test_loader = get_mnist_test_loader(torchvision.datasets.FashionMNIST)

    print("[main] loading digit model")
    digit_model = load_model(
        digit_confusion_run,
        train_S=True,
        replications_per_dim=get_replications_per_dim_from_path(digit_confusion_run),
        device=device,
    )
    print("[main] loading fashion model")
    fashion_model = load_model(
        fashion_confusion_run,
        train_S=True,
        replications_per_dim=get_replications_per_dim_from_path(fashion_confusion_run),
        device=device,
    )

    print("[main] plotting digit twin plot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_accuracies(
        digit_data["accuracies_random"],
        digit_data["replications_random"],
        digit_data["accuracies_trained"],
        digit_data["replications_trained"],
        filename="figs/accuracy_vs_replications_digitmnist_paper.png",
        colors=["blue", "darkblue"],
        yticks=[96.5, 97.0, 97.5, 98.0, 98.5],
        ax=axes[0],
    )
    plot_confusion(
        digit_model,
        digit_test_loader,
        ax=axes[1],
        cmap="Blues",
        class_names=[str(i) for i in range(10)],
    )
    plt.tight_layout()
    plt.savefig("figs/twinplot_accuracy_confusion_digitmnist_paper.png", dpi=300)
    plt.close(fig)
    print("[main] saved digit twin plot")

    fashion_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    print("[main] plotting fashion twin plot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_accuracies(
        fashion_data["accuracies_random"],
        fashion_data["replications_random"],
        fashion_data["accuracies_trained"],
        fashion_data["replications_trained"],
        filename="figs/accuracy_vs_replications_fashionmnist_paper.png",
        colors=["red", "darkred"],
        yticks=[87, 88, 89, 90],
        ax=axes[0],
    )
    plot_confusion(
        fashion_model,
        fashion_test_loader,
        ax=axes[1],
        cmap="Reds",
        class_names=fashion_labels,
    )
    plt.tight_layout()
    plt.savefig("figs/twinplot_accuracy_confusion_fashionmnist_paper.png", dpi=300)
    plt.close(fig)
    print("[main] saved fashion twin plot")

    print("Saved:")
    print("- figs/twinplot_accuracy_confusion_digitmnist_paper.png")
    print("- figs/twinplot_accuracy_confusion_fashionmnist_paper.png")


if __name__ == "__main__":
    main()
