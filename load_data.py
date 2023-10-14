import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision

class LoadData:
    def __init__(self,
                 root="./data",
                 batch_size=64,
                 test_size=0.2,
                 random_seed=42):
        self.root = root
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_seed = random_seed


        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        
        self.full_dataset = torchvision.datasets.STL10(
            root=root,
            split="train",
            transform=self.transform,
            download=True,
        )

        self._split_dataset()

    def _split_dataset(self):
        train_indices, val_indices = train_test_split(
            list(range(len(self.full_dataset))), test_size=self.test_size, random_state=self.random_seed
        )
        self.train_subset = Subset(self.full_dataset, train_indices)
        self.val_subset = Subset(self.full_dataset, val_indices)

    def get_dataloader(self):
        train_loader = DataLoader(self.train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_subset, batch_size=self.batch_size, shuffle=False)
        return train_loader


