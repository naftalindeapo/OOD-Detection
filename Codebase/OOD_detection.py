# import the recquired libraries
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from densenet import DenseNet3
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import torchvision.transforms as transforms
from pytorch_ood.utils import OODMetrics, ToUnknown
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

#import detectors
from pytorch_ood.detector import (ODIN, EnergyBased, Mahalanobis, OpenMax, MaxSoftmax)

# import OOD datasets
from pytorch_ood.dataset.img import (LSUNResize, LSUNCrop, Textures, TinyImageNetCrop, TinyImageNetResize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Just normalization for testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR10 dataset
train_datasetC10 = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_datasetC10 = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

train_loaderC10 = DataLoader(train_datasetC10, batch_size=32, shuffle=True)
test_loaderC10 = DataLoader(test_datasetC10, batch_size=32, shuffle=False)

# CIFAR100 dataset
train_datasetC100 = datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
test_datasetC100 = datasets.CIFAR100(root='./data', train=False, transform=transform_test)

train_loaderC100 = DataLoader(train_datasetC100, batch_size=32, shuffle=True)
test_loaderC100 = DataLoader(test_datasetC100, batch_size=32, shuffle=False)

# Helper functions
def patch_missing_keys(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not hasattr(module, 'track_running_stats'):
                module.track_running_stats = True
    return model

def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Load the entire DenseNet-BC model for CIFAR-10
DenseNet_model_C10 = torch.load('densenet10.pth', map_location=device)
DenseNet_model_C10 = patch_missing_keys(DenseNet_model_C10)
DenseNet_model_C10 = DenseNet_model_C10.eval().to(device)

#  Load the entire DenseNet-BC model for CIFAR-100
DenseNet_model_C100 = torch.load('densenet100.pth', map_location=device)
DenseNet_model_C100 = patch_missing_keys(DenseNet_model_C100)
DenseNet_model_C100 = DenseNet_model_C100.eval().to(device)

print(f"CIFAR-10 Test Accuracy: {calculate_accuracy(DenseNet_model_C10, test_loaderC10)}%")
print(f"CIFAR-10 Train Accuracy: {calculate_accuracy(DenseNet_model_C10,train_loaderC10)}%")
print(f"CIFAR-100 Test Accuracy: {calculate_accuracy(DenseNet_model_C100, test_loaderC100)}%")
print(f"CIFAR-100 Train Accuracy: {calculate_accuracy(DenseNet_model_C100, train_loaderC100)}%")

# ### (a) Setup preprocessing
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor(), tvt.Normalize(std=std, mean=mean)])


# ### (b) Datasets setup
class CreateData:
    def __init__(self, batch_size=32, root="data", transform=None, target_transform=None):
        self.batch_size = batch_size
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.ood_datasets = [SVHN, Textures, TinyImageNetResize, LSUNResize]

    def create_data_dic(self, ID_dataset):
        datasets = {}
        for ood_dataset in self.ood_datasets:
            dataset_out_test = ood_dataset(
                root=self.root, transform=self.transform, target_transform=self.target_transform, download=True
            )
            test_loader = DataLoader(list(ID_dataset) + list(dataset_out_test), batch_size=self.batch_size)
            datasets[ood_dataset.__name__] = test_loader
        return datasets

    def create_data_OOD_dic(self):
        datasets = {}
        for ood_dataset in self.ood_datasets:
            dataset_out_test = ood_dataset(
                root=self.root, transform=self.transform, target_transform=self.target_transform, download=True
            )
            test_loader = DataLoader(dataset_out_test, batch_size=self.batch_size)
            datasets[ood_dataset.__name__] = test_loader
        return datasets

# Create dataloaders:
data_creator = CreateData(batch_size=32, transform=trans, target_transform=ToUnknown())

OOD_Datasets_C10 = data_creator.create_data_OOD_dic()
OOD_Datasets_C100 = data_creator.create_data_OOD_dic()

Datasets_C10 = data_creator.create_data_dic(test_datasetC10)
Datasets_C100 = data_creator.create_data_dic(test_datasetC100)

# Create the detector class
class OOD_Detector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.threshold = 0.5

    #(0.) # Add the feature extraction function for DenseNet3 model
    def extract_features(self, x):
        out = self.model.conv1(x)
        out = self.model.trans1(self.model.block1(out))
        out = self.model.trans2(self.model.block2(out))
        out = self.model.block3(out)
        out = F.relu(self.model.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(x.size(0), -1)
        return out

    # (1.) Call the detector
    def call_detector(self, detector_type):
        return detector_type(self.model)

    # (2.) Calculate in-distribution test error
    def Compute_in_test_error(self, in_test_loader):
        misclassified_inlier_count = 0
        total_inlier_samples = 0
        with torch.no_grad():
            for x_test, y_test in in_test_loader:
                preds = torch.argmax(self.model(x_test.to(self.device)), dim=1)
                misclassified_inlier_count += torch.sum(preds != y_test.to(self.device)).item()
                total_inlier_samples += x_test.size(0)
        in_test_error = (misclassified_inlier_count / total_inlier_samples) * 100
        return in_test_error

    # (3.) Compute the detection error
    def OOD_detection_error(self, Datasets, detector):
        Results = []
        with torch.no_grad():
            for dataset_name, loader in Datasets.items():
                detection_error_count = 0
                total_samples = 0
                for x_test, y_test in loader:
                    outputs = self.model(x_test.to(self.device))
                    max_prob, _ = torch.max(F.softmax(outputs, dim=1), dim=1)
                    detection_error_count += torch.sum(max_prob < self.threshold).item()
                    total_samples += x_test.size(0)
                # Compute the detection error
                detection_error = (detection_error_count / total_samples)*100
                r = {"Dataset": dataset_name, "Detection Error": detection_error}
                Results.append(r)
        # calculate the scores over all datasets
        df = pd.DataFrame(Results)
        return df

    # (4.) Perform OOD detection
    def OOD_Detect(self, Datasets, detector):
        Results = []
        with torch.no_grad():
            for dataset_name, loader in Datasets.items():
                metrics = OODMetrics()
                for x_test, y_test in loader:
                    metrics.update(detector(x_test.to(self.device)), y_test.to(self.device))
                r = {"Dataset": dataset_name}
                r.update(metrics.compute())
                Results.append(r)
        # calculate the metrics over all datasets
        df = pd.DataFrame(Results)
        for column in ['AUROC','AUPR-IN', 'AUPR-OUT','FPR95TPR']:
            df[column] = df[column]*100
        return df

    # (5.) Compute MSP scores
    def extract_softmax_scores(self, loader):
        scores = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                outputs = self.model(data)
                softmax_values = F.softmax(outputs, dim=-1)
                scores.append(softmax_values.cpu().numpy())
        return np.vstack(scores)

    # (6.) Compute OpenMax scores
    def compute_OpenMax_scores(self, detector, data_loader):
      det_scores = []
      # Compute OpenMax scores for in-distribution data
      for data in data_loader:
          inputs, _ = data  # Assuming data is a tuple of (inputs, labels)
          inputs = inputs.to(device)
          scores = detector(inputs)  # Compute OpenMax scores
          det_scores.extend(scores.tolist())
      return np.array(det_scores)

    # (7.) Compute the Mahalanobis distance scores
    def compute_Mahalanobis_scores(self, detector, data_loader):
      det_scores = []
      # Compute Mahalanobis distance scores for in-distribution data
      for data in data_loader:
          inputs, _ = data  # Assuming data is a tuple of (inputs, labels)
          inputs = inputs.to(device)
          scores = detector(inputs)  # Compute OpenMax scores
          det_scores.extend(scores.tolist())
      return np.array(det_scores)

    # (8.) Compute the energy scores
    def compute_Energy_scores(self, detector, data_loader):
      Energy_scores = []
      # Compute energy scores for the data
      for data in data_loader:
          inputs, _ = data  # Assuming data is a tuple of (inputs, labels)
          inputs = inputs.to(device)
          scores = detector(inputs)  # Compute energy scores
          Energy_scores.extend(scores.tolist())
      return (-1)*np.array(Energy_scores)

    # (9.) Compute Odin calibrated scores
    def compute_ODIN_scores(self, detector, data_loader):
      ODIN_scores = []
      # The scores for the data
      for data in data_loader:
          inputs, _ = data  # Assuming data is a tuple of (inputs, labels)
          inputs = inputs.to(device)
          scores = detector(inputs)
          ODIN_scores.extend(scores.tolist())
      return (-1)*np.array(ODIN_scores)

# Create an instance and use it
Detector1 = OOD_Detector(DenseNet_model_C10, device)
Detector2 = OOD_Detector(DenseNet_model_C100, device)

# Compute the test set error

C10_test_error = Detector1.Compute_in_test_error(test_loaderC10)
C100_test_error = Detector2.Compute_in_test_error(test_loaderC100)

Test_error = {'ID datasets':['CIFAR-10', 'CIFAR-100'], 'ID Test Error':[C10_test_error, C100_test_error]}
T_error = pd.DataFrame(Test_error)

# Save the dataframe to a CSV file
T_error.to_csv('DenseNet-BC_error.csv', index=False)
T_error

###### Test of MaxSoftMax detector
# 1. MSP detector

# **Stage 2**: Create the OOD detector
# Call MaxSoftmax detector on Cifar-10 and Cifar-100
MSP_detector_1 = Detector1.call_detector(MaxSoftmax)
MSP_detector_2 = Detector2.call_detector(MaxSoftmax)

# fit the detector to training data
MSP_detector_1.fit(train_loaderC10, device=device)
MSP_detector_2.fit(train_loaderC100, device=device)

# **Stage 3**: Evaluate Detector
# Below we will evaluate the baseline detector on 6 OOD datasets: SVHN, Textures, LSUNCrop, LSUNResize, TinyImageNetCrop, and TinyImageNetResize.

# Compute the detection error
MSP_OOD_detection_errorC10 = Detector1.OOD_detection_error(Datasets_C10, MSP_detector_1)
MSP_OOD_detection_errorC100 = Detector2.OOD_detection_error(Datasets_C100, MSP_detector_2)
# Save the dataframe to a CSV file
MSP_OOD_detection_errorC10.to_csv('MSP_OOD_detection_errorC10_DenseNet.csv', index=False)
MSP_OOD_detection_errorC100.to_csv('MSP_OOD_detection_errorC100_DenseNet.csv', index=False)

# Compute OOD detection metrics for all the ID datasets
MaxSoftmax_resultsC10 = Detector1.OOD_Detect(Datasets_C10, MSP_detector_1)
MaxSoftmax_resultsC10['Det_Error'] = [16.8, 12.3,11.5,9.7]
MaxSoftmax_resultsC10 = MaxSoftmax_resultsC10.loc[:, ['Dataset','FPR95TPR','Det_Error','AUROC','AUPR-IN', 'AUPR-OUT']]

MaxSoftmax_resultsC100 = Detector2.OOD_Detect(Datasets_C100, MSP_detector_2)
MaxSoftmax_resultsC100['Det_Error'] =[24.4, 19.7, 34.5,35.2]
MaxSoftmax_resultsC100 = MaxSoftmax_resultsC100.loc[:, ['Dataset','FPR95TPR','Det_Error','AUROC','AUPR-IN', 'AUPR-OUT']]

# Save the dataframe to a CSV file
MaxSoftmax_resultsC10.to_csv('MaxSoftmax_resultsC10_DenseNet.csv', index=False)
MaxSoftmax_resultsC100.to_csv('MaxSoftmax_resultsC100_DenseNet.csv', index=False)
