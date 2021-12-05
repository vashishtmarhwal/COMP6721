import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sn
import pandas as pd
import torch
import torchvision
from torchvision.datasets import ImageFolder
import sklearn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import warnings
from random import randint
import sys
warnings.filterwarnings('ignore')

image_path = "./FinalDataset"

class convolutional_neural_network(nn.Module):
    def __init__(self):
        super(convolutional_neural_network, self).__init__()
        self.clayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.complete_connected_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(32768, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 4)
        )

    def forward(self, data):
        data = self.clayer(data)
        size = data.size(0)
        data = data.view(size, -1)
        data = self.complete_connected_layer(data)
        return data

class ModelTrainer:
    def __init__(self):
        self.model = convolutional_neural_network()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.8)
        self.loss_fn = nn.CrossEntropyLoss()

    def validate(self, loader, metric_flag):
        self.model.eval()
        correct = 0
        total = 0
        acc_percent = 0
        with torch.no_grad():
            for imgs, labels in loader:
                outputs = self.model(imgs)
                predicted = (torch.max(outputs, dim=1)[1]).numpy()
                total += labels.shape[0]
                correct += int((predicted == labels.numpy()).sum())
                if metric_flag:
                    print(metrics.confusion_matrix(predicted, labels))
                    cm_array = metrics.confusion_matrix(predicted, labels)
                    df_cm = pd.DataFrame(cm_array, index = [i for i in ['Cloth', 'FFP2', 'Surgical',"Without_Mask"]], columns = [i for i in ['Cloth', 'FFP2', 'Surgical',"Without_Mask"]])
                    plt.figure(figsize = (10,7))
                    sn.heatmap(df_cm, annot=True)
                    plt.xlabel("Predicted Values")
                    plt.ylabel("Actual Values")
                    plt.savefig('ConfusionMatrix.png')
                    print(metrics.classification_report(predicted, labels, target_names=['Cloth', 'FFP2', 'Surgical',"Without_Mask"]))

        acc_percent = (correct / total) * 100    
        self.model.train()
        return acc_percent

    def training_loop(self, loader, n_epochs):
        print("Entering Training Loop")
        # Track loading using tqdm
        for epoch in range(1, n_epochs + 1):
            for imgs, labels in loader:
                result = self.model(imgs)
                calc_loss = self.loss_fn(result, labels)
                self.optimizer.zero_grad()
                calc_loss.backward()
                self.optimizer.step()
            print("Epoch {} accuracy: {:.3f}".format(epoch , self.validate(loader, False)))
        return 


def loadImages(path):
    '''Put files into lists and return them as one list with all images 
     in the folder'''
    image_files = sorted([os.path.join(path, 'cloth', file)
                          for file in os.listdir(path + "/cloth")
                          if file.endswith('.jpg')])
   
    image_files += sorted([os.path.join(path, 'cloth', file)
                          for file in os.listdir(path + "/cloth")
                          if file.endswith('.PNG')])
    
    image_files += sorted([os.path.join(path, 'cloth', file)
                          for file in os.listdir(path + "/cloth")
                          if file.endswith('.jpeg')])
    
    image_files += sorted([os.path.join(path, 'ffp2', file)
                          for file in os.listdir(path + "/ffp2")
                          if file.endswith('.jpg')])
    
    image_files += sorted([os.path.join(path, 'ffp2', file)
                          for file in os.listdir(path + "/ffp2")
                          if file.endswith('.PNG')])
    
    image_files += sorted([os.path.join(path, 'ffp2', file)
                          for file in os.listdir(path + "/ffp2")
                          if file.endswith('.jpeg')])
    
    image_files += sorted([os.path.join(path, 'surgical', file)
                          for file in os.listdir(path + "/surgical")
                          if file.endswith('.jpg')])
    
    image_files += sorted([os.path.join(path, 'surgical', file)
                          for file in os.listdir(path + "/surgical")
                          if file.endswith('.PNG')])
    
    image_files += sorted([os.path.join(path, 'surgical', file)
                          for file in os.listdir(path + "/surgical")
                          if file.endswith('.jpeg')])
    
    image_files += sorted([os.path.join(path, 'without_mask', file)
                          for file in os.listdir(path + "/without_mask")
                          if file.endswith('.jpg')])
    
    image_files += sorted([os.path.join(path, 'without_mask', file)
                          for file in os.listdir(path + "/without_mask")
                          if file.endswith('.PNG')])
    
    image_files += sorted([os.path.join(path, 'without_mask', file)
                          for file in os.listdir(path + "/without_mask")
                          if file.endswith('.jpeg')])
    return image_files

names = ['Cloth', 'FFP2', 'Surgical',"Without_Mask"]
N = []
N.append(len(os.listdir(image_path + "/cloth")))
N.append(len(os.listdir(image_path + "/ffp2")))
N.append(len(os.listdir(image_path + "/surgical")))
N.append(len(os.listdir(image_path + "/without_mask")))

def load_data(path):
    reshape_size = torchvision.transforms.Resize((128,128))
    data_type = torchvision.transforms.ToTensor()
    normalized_metrics = torchvision.transforms.Normalize(
        (0.5, 0.5, 0.5), 
        (0.5, 0.5, 0.5)
    )
    return ImageFolder(root = path,transform = torchvision.transforms.Compose([reshape_size, data_type, normalized_metrics]))

def split_data(dataset):
    # The training data is 75% of the full data set and the testing data is 25% of the original data.
    train_d, test_d = train_test_split(dataset,
                                       test_size=0.25, 
                                       random_state=30
                                      )
    return train_d, test_d

def train_dataloader(dataset):
    return DataLoader(dataset=dataset, 
                      num_workers=2, 
                      shuffle=True,
                      batch_size=4
                     )

def test_dataloader(dataset):
    return DataLoader(dataset=dataset,
                      num_workers=2, 
                      shuffle=True,
                      batch_size=500
                     )

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def label_gen(d):
    labels = ['Cloth', 'FFP2', 'Surgical',"Without_Mask"]
    return labels[d]

def predictSingle():
    if os.path.isfile('./k_cross_CNN.pt'):
        index = randint(0,3)
        test_img_path = "./testImage"
        dataset = load_data(test_img_path)
        img_loader = DataLoader(dataset=dataset, 
                      num_workers=2, 
                      shuffle=True,
                      batch_size=4
                     )
        model =  torch.load('./k_cross_CNN.pt')
        print(model)
        print("Loading Successsful")
        dataiter = iter(img_loader)
        images, labels = dataiter.next()
        
        model_out = model(images[index].unsqueeze(0))
        predicted_class = label_gen(model_out.argmax(dim=1).numpy()[0])
        true_label = label_gen(labels[index].numpy())
        print('Prediction: %s - Actual target: %s'%(predicted_class, true_label))
        imshow(images[index], title = 'Prediction: %s - Actual target: %s'%(predicted_class, true_label))
        plt.show()
    else:
        print("No model found")

def trainNew():
    dataset = load_data(image_path)
    print("Training a new one")
    torch.manual_seed(42)
    num_epochs = 20
    k = 10
    #splits = KFold(n_splits = k, random_state = 42)
    splits = StratifiedShuffleSplit(n_splits = k, test_size = 0.2, random_state = 42)
    kfold_acc = []

    for fold, (train_id, val_id) in enumerate(splits.split(np.arange(len(dataset)), dataset.targets)):
        print("Fold {}".format(fold+1))

        tr_sampler = SubsetRandomSampler(train_id)
        te_sampler = SubsetRandomSampler(val_id)
        tr_loader = DataLoader(dataset, num_workers=2, batch_size = 4, sampler = tr_sampler)
        te_loader = DataLoader(dataset, num_workers = 2, batch_size = 500, sampler = te_sampler)
    
        model_obj = ModelTrainer()
        model_obj.training_loop(tr_loader, num_epochs)
        testdata_accuracy = model_obj.validate(te_loader, True)
        kfold_acc.append(testdata_accuracy)
        print("Model accuracy for test dataset on Fold %d: %d"%(fold+1, testdata_accuracy))

    print("Average Model Accuracy is: ",np.mean(kfold_acc))
    torch.save(model_obj.model, "k_cross_CNN.pt")

def validate_bias(model, loader, type, metric_flag=1):
    model.eval()
    correct = 0
    total = 0
    acc_percent = 0
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            predicted = (torch.max(outputs, dim=1)[1]).numpy()
            total += labels.shape[0]
            correct += int((predicted == labels.numpy()).sum())
            if metric_flag:
                print(metrics.confusion_matrix(predicted, labels))
                cm_array = metrics.confusion_matrix(predicted, labels)
                df_cm = pd.DataFrame(cm_array, index = [i for i in ['Cloth', 'FFP2', 'Surgical',"Without_Mask"]], columns = [i for i in ['Cloth', 'FFP2', 'Surgical',"Without_Mask"]])
                plt.figure(figsize = (10,7))
                sn.heatmap(df_cm, annot=True)
                plt.xlabel("Predicted Values")
                plt.ylabel("Actual Values")
                plt.savefig('ConfusionMatrix'+type+'.png')
                print(metrics.classification_report(predicted, labels, target_names=['Cloth', 'FFP2', 'Surgical',"Without_Mask"]))

    acc_percent = (correct / total) * 100
    return acc_percent


def biasTest():
    model =  torch.load('./k_cross_CNN.pt')
    print("""
    Type of Bias to test:
    1. Gender
    2. Race
    """)
    biastype = int(input())
    if biastype == 1:
        data = "./bias_dataset/gender"
        ##For Male
        path_male = data + "/male"
        dataset = load_data(path_male)
        train_b, test_b = split_data(dataset)
        test_loader_bias = test_dataloader(test_b)
        acc = validate_bias(model, test_loader_bias, "male")
        print("Accuracy for Test Data containing only Males - ", acc)
        ##For Female
        path_female = data + "/female"
        dataset = load_data(path_female)
        train_b, test_b = split_data(dataset)
        test_loader_bias = test_dataloader(test_b)
        acc = validate_bias(model, test_loader_bias, "female")
        print("Accuracy for Test Data containing only Females - ", acc)
    else:
        data = "./bias_dataset/race"
        ##For Eastern
        path_male = data + "/east"
        dataset = load_data(path_male)
        train_b, test_b = split_data(dataset)
        test_loader_bias = test_dataloader(test_b)
        acc = validate_bias(model, test_loader_bias, "east")
        print("Accuracy for Test Data containing only Eastern - ", acc)
        ##For Western
        path_female = data + "/west"
        dataset = load_data(path_female)
        train_b, test_b = split_data(dataset)
        test_loader_bias = test_dataloader(test_b)
        acc = validate_bias(model, test_loader_bias, "west")
        print("Accuracy for Test Data containing only Western - ", acc)





def userInput(x):
    if x == 1:
        biasTest()
    if x == 2:
        print("Predicting Single")
        predictSingle()
    if x == 3:
        trainNew()
    if x == 4:
        return -1

if __name__ == '__main__':
    dataset = load_data(image_path)
    plt.figure(figsize=(9, 3))
    plt.bar(names, N)
    plt.savefig('DatasetStat.png')

    #Send dataset to function

    print("""
    1. Bias Tracking 
    2. Predict for Single Image
    3. Train New Model
    4. Exit
    """)
    userIn = int(sys.argv[1])
    print(userIn)
    userInput(userIn)

