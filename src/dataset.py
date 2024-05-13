# Import libraries
import torch, torchvision, os, pandas as pd
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn; from PIL import Image
from torchvision import transforms as T; from glob import glob
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, Flowers102
# Set manual seed
torch.manual_seed(2021)

def download_data(root, transformations, ds_name): 
    
    if ds_name == "cifar10": return (CIFAR10(root, download=True, train=True, transform=transformations), CIFAR10(root, download=True, train=False, transform=transformations))
    elif ds_name == "cifar100": return (CIFAR100(root, download=True, train=True, transform=transformations), CIFAR100(root, download=True, train=False, transform=transformations))
    elif ds_name == "mnist": return (MNIST(root, download=True, train=True, transform=transformations), MNIST(root, download=True, train=False, transform=transformations))
    elif ds_name == "flowers": return (Flowers102(root, download=True, split="train", transform=transformations), Flowers102(root, download=True, split="val", transform=transformations), Flowers102(root, download=True, split="test", transform=transformations))

def get_dls(root, transformations, bs, ds_name, split = [0.8, 0.1], ns = 4):
    
    if ds_name in ["cifar10", "cifar100", "mnist"]:
        
        ds, ts_ds = download_data(root, transformations, ds_name)
        classes = ds.classes
        all_len = len(ds); tr_len = int(all_len * split[0]); val_len = all_len - tr_len
        tr_ds, val_ds = random_split(dataset = ds, lengths = [tr_len, val_len])
    elif ds_name == "flowers":
        classes = list(pd.read_csv("data/flowers-102/cls_names.csv")["Name"])
        tr_ds, val_ds, ts_ds = download_data(root, transformations, ds_name)
        
    print(f"There are {len(tr_ds)}, {len(val_ds)}, {len(ts_ds)} data in the train, validation, and test datasets, respectively.\n")
    
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, classes