import os
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_non_text_files(directory):
    return [os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if not file.endswith((".txt", ".ipynb"))]


DATA_DIR = "./data"

class ArrayDataset(Dataset):
    def __init__(self, filenames, device, target_shape=(450, 55, 64, 64)):
        filesize = np.prod(target_shape)

        def load_and_print(fn): # Legendary nested func
            start = time.time()
            t = torch.ShortTensor(torch.ShortStorage.from_file(fn, size=filesize, shared=True))
            end = time.time()
            print(f"Loading file: {fn} in {round(end-start, 5)} sec")
            return t

        self.array = torch.cat([load_and_print(fn) for fn in filenames])
        # ^^^ init dataset with pytorch memory hack inside list concatenation (looks fancy at least I hope this works)
        self.array = self.array.reshape((target_shape[0]*len(filenames), target_shape[1], target_shape[2], target_shape[3]))
        self.array = self.array.to(device)
        # Reshape into 4D array

        # Get approximation of mean and std
        if self.array.shape[0] > 5:
            self.mean = self.array[:5].flatten().float().mean()
            self.std = self.array[:5].flatten().float().std()
        else:
            self.mean, self.std = 1, 1

        # Transform compose to apply to scans
        self.transform = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, index):
        scan = self.array[index]
        transformed_scan = self.transform(scan.float())
        print(transformed_scan.shape)
        return transformed_scan


def create_dataset(num_files: int, device):
    target_shape = (450, 55, 64, 64)
    filenames = get_non_text_files(DATA_DIR)
    dataset = ArrayDataset(filenames[:num_files], device)
    return dataset


