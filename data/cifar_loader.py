import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def get_cifar_dataloaders(
    dataset_name: str = "CIFAR-10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    force_synthetic: bool = False
):
    """
    Creates and returns train and test DataLoaders for CIFAR-10, CIFAR-100, Fashion-MNIST, or SVHN.
    Includes automatic downloading and instant synthetic dataset fallback.
    """
    os.makedirs(data_dir, exist_ok=True)
    dname = dataset_name.upper().replace("_", "-")

    if force_synthetic or os.environ.get("USE_SYNTHETIC") == "1":
        print(f"[Data] Using fast synthetic dataset for {dname}.")
        return _make_synthetic_dataset(dname, batch_size, num_workers)

    # 1. CIFAR-100 Setup
    if dname == "CIFAR-100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        try:
            train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
            test_ds = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
            print(f"[Data] Successfully loaded real CIFAR-100 ({len(train_ds)} train, {len(test_ds)} val samples).")
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            )
        except Exception as e:
            print(f"[Data] CIFAR-100 download failed ({e}). Using synthetic dataset.")

    # 2. Fashion-MNIST Setup
    elif dname in ["FASHION-MNIST", "FASHIONMNIST"]:
        transform = transforms.Compose([
            transforms.Resize((32, 32)), # Resize 28x28 to 32x32 for CNN compatibility
            transforms.Grayscale(num_output_channels=3), # Convert 1-channel to 3-channel RGB
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        try:
            train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
            test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
            print(f"[Data] Successfully loaded real Fashion-MNIST ({len(train_ds)} train, {len(test_ds)} val samples).")
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            )
        except Exception as e:
            print(f"[Data] Fashion-MNIST download failed ({e}). Using synthetic dataset.")

    # 3. SVHN Setup
    elif dname == "SVHN":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        try:
            train_ds = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
            test_ds = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
            print(f"[Data] Successfully loaded real SVHN ({len(train_ds)} train, {len(test_ds)} val samples).")
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            )
        except Exception as e:
            print(f"[Data] SVHN download failed ({e}). Using synthetic dataset.")

    # 4. Default CIFAR-10 Setup
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        try:
            train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
            test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
            print(f"[Data] Successfully loaded real CIFAR-10 ({len(train_ds)} train, {len(test_ds)} val samples).")
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            )
        except Exception as e:
            print(f"[Data] CIFAR-10 download failed ({e}). Using synthetic dataset.")

    # 5. Synthetic Fallback Dataset (3,000 train samples)
    return _make_synthetic_dataset(dname, batch_size, num_workers)

def _make_synthetic_dataset(dname: str, batch_size: int, num_workers: int):
    num_train = 3000
    num_test = 600
    num_classes = 100 if dname == "CIFAR-100" else 10

    torch.manual_seed(42)
    x_train = torch.randn(num_train, 3, 32, 32)
    y_train = torch.randint(0, num_classes, (num_train,))
    for c in range(num_classes):
        mask = (y_train == c)
        x_train[mask] += (c * 0.15)

    x_test = torch.randn(num_test, 3, 32, 32)
    y_test = torch.randint(0, num_classes, (num_test,))
    for c in range(num_classes):
        mask = (y_test == c)
        x_test[mask] += (c * 0.15)

    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

# Alias for generic multi-dataset loader
get_dataset_dataloaders = get_cifar_dataloaders
