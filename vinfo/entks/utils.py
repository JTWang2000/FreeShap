import logging
import pathlib
import sys
import time
import random

import torch
import torch.nn as nn

import datasets as hf_datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from .nlpmodels import SentenceClassifier

from .dataset.FGVCAircraft import FGVCAircraft
from .dataset.Food101 import Food101
from .dataset.Flowers102 import Flowers102

# From https://stackoverflow.com/a/1094933
def humanize_units(size, unit="B"):
    for prefix in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if size < 1024.0 or prefix == "Pi":
            break
        size /= 1024.0
    return f"{size:.1f}{prefix}"


def init_torch(allow_tf32=False, benchmark=False, deterministic=True, verbose=False):
    # Disable tf32 in favor of more accurate gradients
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    # Benchmarking can lead to non-determinism
    torch.backends.cudnn.benchmark = benchmark

    # Ensure repeated gradient calculations are consistent
    torch.backends.cudnn.deterministic = deterministic

    if verbose:
        logging.info(f"{torch.backends.cuda.matmul.allow_tf32 = }")
        logging.info(f"{torch.backends.cudnn.allow_tf32 = }")
        logging.info(f"{torch.backends.cudnn.benchmark = }")
        logging.info(f"{torch.backends.cudnn.deterministic = }")


def init_logging(handle, logdir):
    if logdir is not None:
        logdir = pathlib.Path(logdir)
        logdir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        filename = logdir / f"{handle}-{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(filename=filename),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info(f"Logging to {filename}")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG,
            stream=sys.stdout,
        )


def load_model(name, args):
    rng_state = torch.get_rng_state()
    if name == "resnet-18_init":
        torch.manual_seed(438)
        model = models.resnet18()
        model.fc = nn.Linear(512, 1)
    elif name == "resnet-18_pretrained":
        torch.manual_seed(438)
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 1)
    elif name == "resnet-34_init":
        torch.manual_seed(438)
        model = models.resnet34()
        model.fc = nn.Linear(512, 1)
    elif name == "resnet-34_pretrained":
        torch.manual_seed(438)
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 1)
    elif name == "resnet-50_init":
        torch.manual_seed(438)
        model = models.resnet50()
        model.fc = nn.Linear(2048, 1)
    elif name == "resnet-50_pretrained":
        torch.manual_seed(438)
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 1)
    elif name == "resnet-101_init":
        torch.manual_seed(438)
        model = models.resnet101()
        model.fc = nn.Linear(2048, 1)
    elif name == "resnet-101_pretrained":
        torch.manual_seed(438)
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, 1)
    elif name == "resnext-101-32x8d_init":
        torch.manual_seed(438)
        model = models.resnet101_32x8d()
        model.fc = nn.Linear(2048, 1)
    elif name == "resnext-101-32x8d_pretrained":
        torch.manual_seed(438)
        model = models.resnet101_32x8d(pretrained=True)
        model.fc = nn.Linear(2048, 1)
    elif name == "efficientnet-b7_init":
        torch.manual_seed(438)
        model = models.efficientnet_b7()
        model.classifier[1] = nn.Linear(2560, 1)
    elif name == "efficientnet-b7_pretrained":
        torch.manual_seed(438)
        model = models.efficientnet_b7(pretrained=True)
        model.classifier[1] = nn.Linear(2560, 1)
    elif 'bert' in name:
        torch.manual_seed(0)
        model = SentenceClassifier(model_name=name, num_frozen_layers=args.num_frozen_layers)
    else:
        assert False
    torch.set_rng_state(rng_state)

    return model


def load_FakeData():
    transform = transforms.ToTensor()
    dataset = datasets.FakeData(size=250, transform=transform)
    return dataset


def load_CIFAR10(datadir, split):
    root = str(datadir / "CIFAR-10")
    train = split == "train"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = datasets.CIFAR10(root, train=train, transform=transform, download=True)
    return dataset


def load_CIFAR100(datadir, split):
    root = str(datadir / "CIFAR-100")
    train = split == "train"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = datasets.CIFAR100(root, train=train, transform=transform, download=True)
    return dataset


def load_SVHN(datadir, split):
    root = str(datadir / "SVHN")
    mean = [0.4380, 0.4440, 0.4730]
    std = [0.1751, 0.1771, 0.1744]
    transform = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = datasets.SVHN(root, split=split, transform=transform, download=True)
    return dataset


def load_FashionMNIST(datadir, split):
    root = str(datadir / "FashionMNIST")
    train = split == "train"
    mean = [0.2860]
    std = [0.3530]
    transform = [
        transforms.Grayscale(3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = datasets.FashionMNIST(
        root, train=train, transform=transform, download=True
    )
    return dataset


def load_FGVCAircraft(datadir, split):
    root = str(datadir / "FGVCAircraft")
    train = split == "train"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = FGVCAircraft(root, train=train, transform=transform, download=True)
    return dataset


def load_Food101(datadir, split):
    root = str(datadir / "Food-101")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = Food101(root, split=split, transform=transform, download=True)
    return dataset


def load_Flowers102(datadir, split):
    root = str(datadir / "Flowers-102")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = Flowers102(root, split=split, transform=transform, download=True)
    return dataset


def load_sst2(split, args):
    model_name = args.model

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch['sentence'], padding=True, truncation=True)

    dataset = hf_datasets.load_dataset('sst2')
    train_dataset = dataset[split]

    if args.subset and split == 'train':
        num_dp = args.num_dp
        seed = 2023
        random.seed(seed)# Set the seed value to 2023
        seed_numbers = random.sample(range(100), 5)

        dataset = hf_datasets.load_dataset("sst2")
        train_data = dataset['train']
        target_data = train_data.shuffle(seed=seed).select(range(50))
        target_data_ids = target_data['idx']
        train_data = train_data.filter(lambda example: example['idx'] not in target_data_ids)
        
        seed = seed_numbers[0] # try to be consistent
        train_data = train_data.shuffle(seed=seed).select(range(num_dp-1))
        sampled_idx = target_data_ids + train_data['idx']
        print("num data points:", len(sampled_idx))

        train_dataset = dataset['train'].select(sampled_idx)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))

    # Format labels
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return train_dataset



def load_subset(name, split, dataset):
    train = split == "train"
    _, train_begin, train_end, test_begin, test_end = name.split("_")
    subset = (train_begin, train_end) if train else (test_begin, test_end)
    subset_begin, subset_end = map(int, subset)

    assert subset_begin >= 0, f"{subset_begin} < 0"
    assert subset_begin < subset_end, f"{subset_begin} >= {subset_end}"
    assert subset_end <= len(dataset), f"{subset_end} > {len(dataset)}"

    dataset = torch.utils.data.Subset(dataset, range(subset_begin, subset_end))
    return dataset


def load_dataset(datadir, args, split):
    name = args.dataset
    model_name = args.model
    if name == "FakeData":
        dataset = load_FakeData()
    elif name == "CIFAR-10":
        dataset = load_CIFAR10(datadir, split)
    elif name.startswith("CIFAR-10_"):
        dataset = load_CIFAR10(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "CIFAR-100":
        dataset = load_CIFAR100(datadir, split)
    elif name.startswith("CIFAR-100_"):
        dataset = load_CIFAR100(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "SVHN":
        dataset = load_SVHN(datadir, split)
    elif name.startswith("SVHN_"):
        dataset = load_SVHN(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "FashionMNIST":
        dataset = load_FashionMNIST(datadir, split)
    elif name.startswith("FashionMNIST_"):
        dataset = load_FashionMNIST(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "FGVCAircraft":
        dataset = load_FGVCAircraft(datadir, split)
    elif name.startswith("FGVCAircraft_"):
        dataset = load_FGVCAircraft(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "Food-101":
        dataset = load_Food101(datadir, split)
    elif name.startswith("Food-101_"):
        dataset = load_Food101(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "Flowers-102":
        dataset = load_Flowers102(datadir, split)
    elif name.startswith("Flowers-102_"):
        dataset = load_Flowers102(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name.lower() == 'sst2':
        dataset = load_sst2(split, args)
    elif name.lower().startswith('sst2_'): # subset version of sst2
        args.subset = True
        dataset = load_sst2(split, args)
    else:
        assert False

    return dataset


def num_classes_of(name):
    if name == "FakeData":
        num_classes = 0
    elif name == "CIFAR-10" or name.startswith("CIFAR-10_"):
        num_classes = 10
    elif name == "CIFAR-100" or name.startswith("CIFAR-100_"):
        num_classes = 100
    elif name == "SVHN" or name.startswith("SVHN_"):
        num_classes = 10
    elif name == "FashionMNIST" or name.startswith("FashionMNIST_"):
        num_classes = 10
    elif name == "FGVCAircraft" or name.startswith("FGVCAircraft_"):
        num_classes = 102
    elif name == "Food-101" or name.startswith("Food-101_"):
        num_classes = 101
    elif name == "Flowers-102" or name.startswith("Flowers-102_"):
        num_classes = 102
    elif name.lower() == 'sst2':
        num_classes = 2
    else:
        assert False

    return num_classes
