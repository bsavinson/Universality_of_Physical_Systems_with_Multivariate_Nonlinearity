import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from free_space import FreeSpaceModel


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_run_name(args):
    dataset_name = "FashionMNIST" if args.fashion_mnist else "DigitMNIST"
    base = (
        f"{dataset_name}_SGD_PhaseEncoding_"
        f"model_n{args.model_n}_"
        f"replications_per_dim{args.replications_per_dim}_"
        f"lr{args.learning_rate}_"
        f"momentum{args.momentum}_"
        f"batchsize{args.batch_size}_"
        f"trainS{args.train_S}"
    )
    return base


def get_loaders(data_dir, batch_size, use_fashion):
    if use_fashion:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((14, 14)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomErasing(p=0.25),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((14, 14)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_ds = torchvision.datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_ds = torchvision.datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=test_transform
        )
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((14, 14)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_ds = torchvision.datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_ds = torchvision.datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train the Free Space Model.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_S", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fashion_mnist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model_n", type=int, default=14)
    parser.add_argument("--replications_per_dim", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    run_name = args.run_name or build_run_name(args)
    print(f"Run_name: {run_name}")
    print(f"Input arguments: {args}")

    data_dir = "./mnist_data"
    train_loader, test_loader = get_loaders(
        data_dir, args.batch_size, args.fashion_mnist
    )

    if args.model_n < 14:
        raise ValueError("model_n must be at least 14 for MNIST/FashionMNIST.")

    model = FreeSpaceModel(
        model_n=args.model_n,
        train_S=args.train_S,
        replications_per_dim=args.replications_per_dim,
        amplitude_encoding=False,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    best_accuracy = 0.0
    epoch_list = []
    train_losses, test_losses = [], []
    test_accuracies = []

    print(f"Training Free Space Model (train_S={args.train_S})")

    for epoch in range(args.epochs):
        model.train()
        correct = 0
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 50 == 0:
                print(
                    f"Train Epoch: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}",
                    flush=True,
                )

        train_loss /= len(train_loader.dataset)
        accuracy = 100.0 * correct / len(train_loader.dataset)
        print(
            f"\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} "
            f"({accuracy:.0f}%)\n"
        )

        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data)
                test_loss += criterion(output, target).mean().item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        scheduler.step(epoch)

        epoch_list.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        run_summary = {
            "epochs": epoch_list,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
        }
        torch.save(run_summary, run_name + ".pt")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), run_name + "_best_model.pt")

        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
            f"({accuracy:.0f}%)\n"
        )
        print("Best accuracy: ", best_accuracy)

    run_summary = {
        "epochs": epoch_list,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
    }
    torch.save(run_summary, run_name + ".pt")


if __name__ == "__main__":
    main()
