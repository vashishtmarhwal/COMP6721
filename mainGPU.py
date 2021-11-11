import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sn
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import sklearn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')


data_dir = './FinalDataset'

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([#transforms.RandomRotation(30),  # data augmentations are great
                                       #transforms.RandomResizedCrop(224),  # but not in this case of map tiles
                                       #transforms.RandomHorizontalFlip(),
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], # PyTorch recommends these but in this
                                                            [0.229, 0.224, 0.225]) # case I didn't get good results
                                       ])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

    train_data = ImageFolder(datadir, transform=train_transforms)
    test_data = ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)

def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(image.numpy().transpose((1, 2, 0)))

#display the first image in the dataset
dataiter = iter(trainloader)
images, labels = dataiter.next()
display_img(images[0], labels[0])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
