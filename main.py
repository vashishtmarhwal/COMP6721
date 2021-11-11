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
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import warnings
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
            nn.Linear(64 * 32, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 5)
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
    reshape_size = torchvision.transforms.Resize((32,32))
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

def train_dataloarder(dataset):
    return DataLoader(dataset=dataset, 
                      num_workers=2, 
                      shuffle=True,
                      batch_size=4
                     )

def test_dataloarder(dataset):
    return DataLoader(dataset=dataset,
                      num_workers=2, 
                      shuffle=True,
                      batch_size=500
                     )

if __name__ == '__main__':
    dataset = load_data(image_path)
    plt.figure(figsize=(9, 3))
    plt.bar(names, N)
    plt.savefig('DatasetStat.png')

    torch.manual_seed(42)
    num_epochs = 20
    k = 4
    splits = KFold(n_splits = k, shuffle = True, random_state = 42)
    kfold_acc = []


    for fold, (train_id, val_id) in enumerate(splits.split(np.arange(len(dataset)))):
        print("Fold {}".format(fold+1))

        tr_sampler = SubsetRandomSampler(train_id)
        te_sampler = SubsetRandomSampler(val_id)
        tr_loader = train_dataloarder(dataset)
        te_loader = test_dataloarder(dataset)
# =============================================================================
#         tr_loader = DataLoader(dataset, batch_size = 4, sampler = tr_sampler)
#         te_loader = DataLoader(dataset, batch_size = 128, sampler = te_sampler)
# =============================================================================

        
        model_obj = ModelTrainer()
        model_obj.training_loop(tr_loader, num_epochs)
        testdata_accuracy = model_obj.validate(te_loader, True)
        kfold_acc.append(testdata_accuracy)
        print("Model accuracy for test dataset on Fold %d: %d"%(fold+1, testdata_accuracy))

    print("Average Model Accuracy is: ",np.mean(kfold_acc))
    torch.save(model_obj.model, "k_cross_CNN.pt")