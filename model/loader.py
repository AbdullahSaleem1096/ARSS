import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

def get_worker_dataloader(worker_id: int, num_workers: int, batch_size: int = 64, data_dir: str = './data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    num_samples = len(train_dataset)
    # Ensure reproducible split across all workers
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    
    worker_splits = np.array_split(indices, num_workers)
    split = worker_splits[worker_id]
    
    subset = Subset(train_dataset, split.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    return loader
