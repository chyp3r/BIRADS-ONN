import os
import cv2
import time
import random
import numpy as np
from PIL import Image
from datetime import timedelta

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchmetrics.classification import MulticlassRecall,MulticlassPrecision

import scipy.ndimage.morphology
from fastonn import SelfONN2d

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] # VGG16
VGG16_N = [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64, 'M']
VGG16_f = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'] # TRAINABLE VGG16
VGG16_M = [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M'] # MINI VGG16


def invert(img):
    return 255 - img

def gammaCorrection(src, gamma=None):
    src = invert(src)
    
    if(gamma == None):
        mean = np.mean(src)
        if mean > 127:
            src = invert(src)
            mean = 255 - mean
            
        gamma = np.log(mean) / np.log(128)
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

def grey_open(img):
    img = scipy.ndimage.grey_opening(img, 3)
    return img

def grey_close(img):
    img = scipy.ndimage.grey_closing(img, 3)
    return img

def mean_shift(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    spatial_radius = 7
    color_radius = 25
    num_iterations = 5
    
    segmented_image = cv2.pyrMeanShiftFiltering(img_lab, spatial_radius, color_radius, num_iterations)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    
    return segmented_image

def sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(img, None)
    sift_img = cv2.drawKeypoints(img, keypoints, img)
    
    return sift_img

def apply_clahe(img, clipLimit, tileGridSize):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ValidationLossCheckpoint:
    def __init__(self, path='checkpoint.pth.tar'):
        self.best = float('inf')
        self.path = path

    def __call__(self, current, epoch, model, optimizer, criterion):
        if current < self.best:
            self.best = current
            print(f'{"-"*150}\nBest validation loss: {current:.4f}\tSaving best model for epoch {epoch+1}\n{"-"*150}')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                },
                self.path
            )
    
    def reset(self):
        self.best = float('inf')


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.val_losses = []
        self.val_accuracies = []
        self.train_losses = []
        self.train_accuracies = []
    
    def train(self, num_epochs, train_loader, validation_loader, early_stopper = None, checkpoint = None):
        for epoch in range(num_epochs):
            val_loss = 0
            val_acc = 0
            train_loss = 0
            train_acc = 0
            start = time.time()

            for phase in ['train', 'val']:
                running_loss = 0
                num_correct = 0
                num_samples = 0

                if phase == 'train':
                    self.model.train()
                    loader = train_loader
                else:
                    self.model.eval()
                    loader = validation_loader

                for batch_idx, (data, targets) in enumerate(loader):
                    data = data.to(device=self.device)

                    targets = np.asarray(targets)
                    targets = torch.from_numpy(targets)
                    targets = targets.to(device=self.device)

                    self.optimizer.zero_grad()

                    # scores = model(data0, data1, data2, data3)
                    scores = self.model(data)
                    _, predictions = scores.max(1)
                    loss = self.criterion(scores, targets)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * predictions.size(0)
                    num_correct += (predictions == targets).sum()
                    num_samples += predictions.size(0)

                if phase == 'train':
                    train_loss = running_loss / num_samples
                    train_acc = num_correct / num_samples
                else:
                    val_loss = running_loss / num_samples
                    val_acc = num_correct / num_samples

            delta = time.time() - start
            delta = timedelta(seconds=delta)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc.item())
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc.item())
            print(
                f'Epoch: {epoch+1}/{num_epochs}\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}\t\tTime: {delta}\t{time.strftime("%H:%M:%S", time.localtime())}')

            if early_stopper != None:
                if early_stopper.early_stop(val_loss):
                    print('\nEarlystopping...')
                    break
            if checkpoint != None:
                checkpoint(val_loss, epoch, self.model, self.optimizer, self.criterion)


class CustomTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.val_losses = []
        self.val_accuracies = []
        self.train_losses = []
        self.train_accuracies = []
    
    def train(self, num_epochs, train_loader, validation_loader = None, early_stopper = None, checkpoint = None, scheduler = None):
        import sys
        final_result = []
        recall = MulticlassRecall(num_classes=2,average=None).to(device = self.device)
        precision = MulticlassPrecision(num_classes=2,average=None).to(device = self.device)
        for epoch in range(num_epochs):
            val_loss = 0
            val_acc = 0
            train_loss = 0
            train_acc = 0
            start = time.time()
            
            # Training
            running_loss = 0
            num_correct = 0
            num_samples = 0
            self.model.train()
            loader = train_loader

            train_pre = torch.tensor([]).to(device = self.device)
            train_target = torch.tensor([]).to(device = self.device)

            for batch_idx, (data, targets) in enumerate(loader):  
                data = data.to(device=self.device)
                targets = np.asarray(targets)
                targets = torch.from_numpy(targets)
                targets = targets.to(device=self.device)
                
                scores = self.model(data)
                _, predictions = scores.max(1)
                loss = self.criterion(scores, targets)
                loss.backward()
                self.optimizer.step()
                
                self.optimizer.zero_grad()
                train_pre = torch.cat((train_pre,predictions),0)
                train_target = torch.cat((train_target,targets),0)
                
                
                running_loss += loss.item() * predictions.size(0)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
                # print("Train pre:",predictions,"Train target:",targets)
                # sys.stdout.write("\033[F")
                # print("Trian correct",num_correct,"Values",num_samples)
            train_loss = running_loss / num_samples
            train_acc = num_correct / num_samples
                    
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc.item())

            # Validation
            val_pre = torch.tensor([]).to(device = self.device)
            val_target = torch.tensor([]).to(device = self.device)

            if validation_loader is not None:
                running_loss = 0
                num_correct = 0
                num_samples = 0
                self.model.eval()
                loader = validation_loader
                
                for batch_idx, (data, targets) in enumerate(loader): 
                    data = data.to(device=self.device)

                    targets = np.asarray(targets)
                    targets = torch.from_numpy(targets)
                    targets = targets.to(device=self.device)
                    
                    scores = self.model(data)
                    _, predictions = scores.max(1)
                    loss = self.criterion(scores, targets)

                    val_pre = torch.cat((val_pre,predictions),0)
                    val_target = torch.cat((val_target,targets),0)  

                    running_loss += loss.item() * predictions.size(0)
                    num_correct += (predictions == targets).sum()
                    num_samples += predictions.size(0)
                    # print("Val pre:",predictions,"Val target:",targets,"Scores:",scores)
                    # sys.stdout.write("\033[F")
                    # print("Val correct",num_correct,"Values",num_samples)
                val_loss = running_loss / num_samples
                val_acc = num_correct / num_samples
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc.item())
                
            if scheduler is not None:
                scheduler.step(val_loss)

            delta = time.time() - start
            delta = timedelta(seconds=delta)
            value = f'Epoch: {epoch+1}/{num_epochs}\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}\t\tTime: {delta}\t{time.strftime("%H:%M:%S", time.localtime())}'
            print(value)
            print("Train Recall:",recall(train_pre,train_target),"Validation Recall",recall(val_pre,val_target))
            print("Train Precision:",precision(train_pre,train_target),"Validation Precision",precision(val_pre,val_target))
            final_result.append(value)
            if early_stopper is not None:
                if early_stopper.early_stop(val_loss):
                    print('\nEarlystopping...')
                    break
            if checkpoint is not None:
                checkpoint(val_loss, epoch, self.model, self.optimizer, self.criterion)
        for f in final_result:
            print(f)       
            
    def trainSiamese(self, num_epochs, train_loader, validation_loader = None, early_stopper = None, checkpoint = None, scheduler = None):
        for epoch in range(num_epochs):
            val_loss = 0
            val_acc = 0
            train_loss = 0
            train_acc = 0
            start = time.time()
            
            # Training
            running_loss = 0
            num_correct = 0
            num_samples = 0
            self.model.train()
            loader = train_loader
            
            for batch_idx, (data, targets) in enumerate(loader):
                data_0, data_1 = data
                data_0, data_1 = data_0.to(device=self.device), data_1.to(device=self.device)
                
                targets = np.asarray(targets)
                targets = torch.from_numpy(targets)
                targets = targets.to(device=self.device)
                
                score0, score1 = self.model(data_0, data_1)
                
                loss = self.criterion(score0, score1, targets)
                loss.backward()
                self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                running_loss += loss.data[0] * targets.size(0)
                num_samples += targets.size(0)
                

            train_loss = running_loss / num_samples

            self.train_losses.append(train_loss)

            # Validation
            if validation_loader is not None:
                running_loss = 0
                num_correct = 0
                num_samples = 0
                self.model.eval()
                loader = validation_loader
                
                for batch_idx, (data, targets) in enumerate(loader):
                    data_0, data_1 = data
                    data_0, data_1 = data_0.to(device=self.device), data_1.to(device=self.device)

                    targets = np.asarray(targets)
                    targets = torch.from_numpy(targets)
                    targets = targets.to(device=self.device)
                    
                    score0, score1 = self.model(data_0, data_1)
                    loss = self.criterion(score0, score1, targets)
                    
                    running_loss += loss.data[0] * targets.size(0)
                    num_samples += targets.size(0)

                val_loss = running_loss / num_samples

                self.val_losses.append(val_loss)
                
            if scheduler is not None:
                scheduler.step(val_loss)

            delta = time.time() - start
            delta = timedelta(seconds=delta)
            print(
                f'Epoch: {epoch+1}/{num_epochs}\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}\t\tTime: {delta}\t{time.strftime("%H:%M:%S", time.localtime())}')

            if early_stopper is not None:
                if early_stopper.early_stop(val_loss):
                    print('\nEarlystopping...')
                    break
            if checkpoint is not None:
                checkpoint(val_loss, epoch, self.model, self.optimizer, self.criterion)


class DatasetSingleGray(Dataset):
    def __init__(self, dataframe, target_enum, target,
                 root='dataset', input_shape=(224, 224),
                 transform=None, grayscale=False,
                 include_LCC = True, include_LMLO = True, include_RCC = True, include_RMLO = True,
                 include_A = True, include_B = True, include_C = True, include_D = True,
                 gamma_correction = False, clahe = False, grey_open = False, mean_shift = False,             
                 ):
        
        # dataframe preprocessing
        self.dataframe = dataframe
        if not include_LCC:
            self.dataframe = self.dataframe.loc[~self.dataframe['HASTANO'].str.endswith('LCC.jpg')]
        if not include_LMLO:
            self.dataframe = self.dataframe.loc[~self.dataframe['HASTANO'].str.endswith('LMLO.jpg')]
        if not include_RCC:
            self.dataframe = self.dataframe.loc[~self.dataframe['HASTANO'].str.endswith('RCC.jpg')]
        if not include_RMLO:
            self.dataframe = self.dataframe.loc[~self.dataframe['HASTANO'].str.endswith('RMLO.jpg')]
        if not include_A:
            self.dataframe = self.dataframe.loc[self.dataframe['MEME KOMPOZİSYONU'] != 'A']
        if not include_B:
            self.dataframe = self.dataframe.loc[self.dataframe['MEME KOMPOZİSYONU'] != 'B']
        if not include_C:
            self.dataframe = self.dataframe.loc[self.dataframe['MEME KOMPOZİSYONU'] != 'C']
        if not include_D:
            self.dataframe = self.dataframe.loc[self.dataframe['MEME KOMPOZİSYONU'] != 'D']
        self.dataframe = self.dataframe.reset_index(drop=True)
        
        self.target_enum = target_enum
        self.target = target
        self.targets = self.dataframe[self.target].values
        
        self.root = root
        self.input_shape = input_shape

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=18),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ####
                ]
            )
        self.grayscale = grayscale
        
        self.gamma_correction = gamma_correction
        self.clahe = clahe
        self.grey_open = grey_open
        self.mean_shift = mean_shift

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = os.path.join(self.root, row['HASTANO'])
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=self.input_shape)
        
        if self.gamma_correction:
            img = gammaCorrection(img)
        elif np.mean(img) > 127:
            img = invert(img)
            
        if self.clahe:
            img = apply_clahe(img, 1.0, (12, 12))
        
        if self.grey_open:
            img = grey_open(img)
            
        if self.mean_shift:
            img = mean_shift(img)
            
        if row['HASTANO'].endswith('RCC.jpg') or row['HASTANO'].endswith('RMLO.jpg'):
            img = cv2.flip(img, 1)
        
        y_label = self.dataframe.loc[index, self.target]
        
        if not self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = Image.fromarray(img)
        img = self.transform(img)
        
        y_label = torch.tensor(self.target_enum[y_label])
        # y_label = self.target_enum[y_label]
        return img, y_label

    def get_sample_weights(self):
        self.sample_targets = np.append(self.targets, list(self.target_enum.keys()))
        
        unique, counts = np.unique(self.sample_targets, return_counts=True)
        class_weights = dict(zip(unique, counts))
        sample_weights = [0] * len(self.target_enum)
        for i in self.target_enum:
            sample_weights[self.target_enum[i]] = float(
                self.__len__() / class_weights[i])

        return sample_weights


class DatasetMultiGray(Dataset):
    def __init__(self, dataframe, target_enum, target,
                 root='dataset', input_shape=(224, 224),
                 transform=None, grayscale=False,
                 include_LCC = True, include_LMLO = True, include_RCC = True, include_RMLO = True,
                 include_A = True, include_B = True, include_C = True, include_D = True,
                 gamma_correction = False, clahe = False, grey_open = False, mean_shift = False,             
                 ):

        self.dataframe = dataframe
        if not include_LCC:
            self.dataframe = self.dataframe.loc[~self.dataframe['HASTANO'].str.endswith('LCC.jpg')]
        if not include_LMLO:
            self.dataframe = self.dataframe.loc[~self.dataframe['HASTANO'].str.endswith('LMLO.jpg')]
        if not include_RCC:
            self.dataframe = self.dataframe.loc[~self.dataframe['HASTANO'].str.endswith('RCC.jpg')]
        if not include_RMLO:
            self.dataframe = self.dataframe.loc[~self.dataframe['HASTANO'].str.endswith('RMLO.jpg')]
        if not include_A:
            self.dataframe = self.dataframe.loc[self.dataframe['MEME KOMPOZİSYONU'] != 'A']
        if not include_B:
            self.dataframe = self.dataframe.loc[self.dataframe['MEME KOMPOZİSYONU'] != 'B']
        if not include_C:
            self.dataframe = self.dataframe.loc[self.dataframe['MEME KOMPOZİSYONU'] != 'C']
        if not include_D:
            self.dataframe = self.dataframe.loc[self.dataframe['MEME KOMPOZİSYONU'] != 'D']
        self.dataframe = self.dataframe.reset_index(drop=True)
        
        self.target_enum = target_enum
        self.target = target
        self.targets = self.dataframe[self.target].values
        
        self.root = root
        self.input_shape = input_shape

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ####
                ]
            )
        self.grayscale = grayscale
        
        self.gamma_correction = gamma_correction
        self.clahe = clahe
        self.grey_open = grey_open
        self.mean_shift = mean_shift   

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = os.path.join(self.root, str(row['HASTANO']))

        lcc_image_path = os.path.join(path, 'LCC.jpg')
        lmlo_image_path = os.path.join(path, 'LMLO.jpg')
        rcc_image_path = os.path.join(path, 'RCC.jpg')
        rmlo_image_path = os.path.join(path, 'RMLO.jpg')
        
        lcc_img = cv2.imread(lcc_image_path, cv2.IMREAD_GRAYSCALE)
        lmlo_img = cv2.imread(lmlo_image_path, cv2.IMREAD_GRAYSCALE)
        rcc_img = cv2.imread(rcc_image_path, cv2.IMREAD_GRAYSCALE)
        rmlo_img = cv2.imread(rmlo_image_path, cv2.IMREAD_GRAYSCALE)
        
        lcc_img = cv2.resize(lcc_img, dsize=self.input_shape)
        lmlo_img = cv2.resize(lmlo_img, dsize=self.input_shape)
        rcc_img = cv2.resize(rcc_img, dsize=self.input_shape)
        rmlo_img = cv2.resize(rmlo_img, dsize=self.input_shape)
        
        if self.gamma_correction:
            lcc_img = gammaCorrection(lcc_img)
            lmlo_img = gammaCorrection(lmlo_img)
            rcc_img = gammaCorrection(rcc_img)
            rmlo_img = gammaCorrection(rmlo_img)
        elif np.mean(lcc_img) > 127:
            lcc_img = invert(lcc_img)
            lmlo_img = invert(lmlo_img)
            rcc_img = invert(rcc_img)
            rmlo_img = invert(rmlo_img)
            
        if self.clahe:
            lcc_img = apply_clahe(lcc_img, 1.0, (12, 12))
            lmlo_img = apply_clahe(lmlo_img, 1.0, (12, 12))
            rcc_img = apply_clahe(rcc_img, 1.0, (12, 12))
            rmlo_img = apply_clahe(rmlo_img, 1.0, (12, 12))
        
        if self.grey_open:
            lcc_img = grey_open(lcc_img)
            lmlo_img = grey_open(lmlo_img)
            rcc_img = grey_open(rcc_img)
            rmlo_img = grey_open(rmlo_img)
            
        if self.mean_shift:
            lcc_img = mean_shift(lcc_img)
            lmlo_img = mean_shift(lmlo_img)
            rcc_img = mean_shift(rcc_img)
            rmlo_img = mean_shift(rmlo_img)
            
        rcc_img = cv2.flip(rcc_img, 1)
        rmlo_img = cv2.flip(rmlo_img, 1)
        
        y_label = self.dataframe.loc[index, self.target]
        
        if not self.grayscale:
            lcc_img = cv2.cvtColor(lcc_img, cv2.COLOR_GRAY2RGB)
            lmlo_img = cv2.cvtColor(lmlo_img, cv2.COLOR_GRAY2RGB)
            rcc_img = cv2.cvtColor(rcc_img, cv2.COLOR_GRAY2RGB)
            rmlo_img = cv2.cvtColor(rmlo_img, cv2.COLOR_GRAY2RGB)
            
        lcc_img = Image.fromarray(lcc_img)
        lmlo_img = Image.fromarray(lmlo_img)
        rcc_img = Image.fromarray(rcc_img)
        rmlo_img = Image.fromarray(rmlo_img)
        
        lcc_img = self.transform(lcc_img)
        lmlo_img = self.transform(lmlo_img)
        rcc_img = self.transform(rcc_img)
        rmlo_img = self.transform(rmlo_img)
        
        y_label = torch.tensor(self.target_enum[y_label])
        # y_label = self.target_enum[y_label]
        
        return (lcc_img, lmlo_img, rcc_img, rmlo_img), y_label

    def get_sample_weights(self):
        unique, counts = np.unique(self.targets, return_counts=True)
        class_weights = dict(zip(unique, counts))
        sample_weights = [0] * len(self.target_enum)
        for i in self.target_enum:
            sample_weights[self.target_enum[i]] = float(
                self.__len__() / class_weights[i])

        return sample_weights


def getPretrainedModel(checkpoint, device):
    model = torchvision.models.alexnet()
    model.classifier = nn.Sequential(
        nn.Linear(9216, 81),
        nn.ReLU(inplace=True),
        nn.Linear(81, 2),
    )
    
    # Altering model for using with grayscale
    # conv_weight = model.features[0].weight
    # model.features[0].in_channels=1
    # model.features[0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
    model.to(device=device)
    
    model.eval()
    
    return model

def getPretrainedModelComposition(checkpoint, device):
    model = torchvision.models.alexnet()
    model.classifier = nn.Sequential(
        nn.Linear(9216, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(inplace=False),
        nn.Linear(4096, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 4),
    )
    
    # Altering model for using with grayscale
    # conv_weight = model.features[0].weight
    # model.features[0].in_channels=1
    # model.features[0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
    model.to(device=device)
    
    model.eval()
    
    return model

def get_features(models, dataloader):
    features = []
    labels = []
    
    model_0_cc, model_0_mlo, model_12_cc, model_12_mlo, model_45_cc, model_45_mlo = models[0], models[1], models[2], models[3], models[4], models[5]
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs_lcc, inputs_lmlo, inputs_rcc, inputs_rmlo = inputs[0], inputs[1], inputs[2], inputs[3]
            inputs_lcc = inputs_lcc.cuda()
            inputs_lmlo = inputs_lmlo.cuda()
            inputs_rcc = inputs_rcc.cuda()
            inputs_rmlo = inputs_rmlo.cuda()
            targets = targets.cuda()
            
            lcc_0_output = model_0_cc(inputs_lcc)
            lcc_12_output = model_12_cc(inputs_lcc)
            lcc_45_output = model_45_cc(inputs_lcc)
            
            lmlo_0_output = model_0_mlo(inputs_lmlo)
            lmlo_12_output = model_12_mlo(inputs_lmlo)
            lmlo_45_output = model_45_mlo(inputs_lmlo)
            
            rcc_0_output = model_0_cc(inputs_rcc)
            rcc_12_output = model_12_cc(inputs_rcc)
            rcc_45_output = model_45_cc(inputs_rcc)
            
            rmlo_0_output = model_0_mlo(inputs_rmlo)
            rmlo_12_output = model_12_mlo(inputs_rmlo)
            rmlo_45_output = model_45_mlo(inputs_rmlo)
            
            lcc_0_output = lcc_0_output.detach().cpu()
            lcc_12_output = lcc_12_output.detach().cpu()
            lcc_45_output = lcc_45_output.detach().cpu()
            lmlo_0_output = lmlo_0_output.detach().cpu()
            lmlo_12_output = lmlo_12_output.detach().cpu()
            lmlo_45_output = lmlo_45_output.detach().cpu()
            rcc_0_output = rcc_0_output.detach().cpu()
            rcc_12_output = rcc_12_output.detach().cpu()
            rcc_45_output = rcc_45_output.detach().cpu()
            rmlo_0_output = rmlo_0_output.detach().cpu()
            rmlo_12_output = rmlo_12_output.detach().cpu()
            rmlo_45_output = rmlo_45_output.detach().cpu()
            targets = targets.detach().cpu()
            
            lcc_0_output = lcc_0_output.view(lcc_0_output.size(0), -1).numpy()
            lcc_12_output = lcc_12_output.view(lcc_12_output.size(0), -1).numpy()
            lcc_45_output = lcc_45_output.view(lcc_45_output.size(0), -1).numpy()
            lmlo_0_output = lmlo_0_output.view(lmlo_0_output.size(0), -1).numpy()
            lmlo_12_output = lmlo_12_output.view(lmlo_12_output.size(0), -1).numpy()
            lmlo_45_output = lmlo_45_output.view(lmlo_45_output.size(0), -1).numpy()
            rcc_0_output = rcc_0_output.view(rcc_0_output.size(0), -1).numpy()
            rcc_12_output = rcc_12_output.view(rcc_12_output.size(0), -1).numpy()
            rcc_45_output = rcc_45_output.view(rcc_45_output.size(0), -1).numpy()
            rmlo_0_output = rmlo_0_output.view(rmlo_0_output.size(0), -1).numpy()
            rmlo_12_output = rmlo_12_output.view(rmlo_12_output.size(0), -1).numpy()
            rmlo_45_output = rmlo_45_output.view(rmlo_45_output.size(0), -1).numpy()
            
            outputs = np.concatenate((lcc_0_output, lcc_12_output, lcc_45_output, 
                                      lmlo_0_output, lmlo_12_output, lmlo_45_output,
                                      rcc_0_output, rcc_12_output, rcc_45_output,
                                      rmlo_0_output, rmlo_12_output, rmlo_45_output,
                                      ),
                                     axis=1
                                     )
            
            features.append(outputs)
            # features = np.append(features, outputs, axis=0)
            labels.append(targets.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

def get_features_single_model(model, dataloader):
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs_lcc, inputs_lmlo, inputs_rcc, inputs_rmlo = inputs[0], inputs[1], inputs[2], inputs[3]
            inputs_lcc = inputs_lcc.cuda()
            inputs_lmlo = inputs_lmlo.cuda()
            inputs_rcc = inputs_rcc.cuda()
            inputs_rmlo = inputs_rmlo.cuda()
            targets = targets.cuda()
            
            lcc_output = model(inputs_lcc)
            lmlo_output = model(inputs_lmlo)
            rcc_output = model(inputs_rcc)
            rmlo_output = model(inputs_rmlo)
            
            lcc_output = lcc_output.detach().cpu()
            lmlo_output = lmlo_output.detach().cpu()
            rcc_output = rcc_output.detach().cpu()
            rmlo_output = rmlo_output.detach().cpu()
            targets = targets.detach().cpu()
            
            lcc_output = lcc_output.view(lcc_output.size(0), -1).numpy()
            lmlo_output = lmlo_output.view(lmlo_output.size(0), -1).numpy()
            rcc_output = rcc_output.view(rcc_output.size(0), -1).numpy()
            rmlo_output = rmlo_output.view(rmlo_output.size(0), -1).numpy()
            
            outputs = np.concatenate((lcc_output,
                                      lmlo_output,
                                      rcc_output,
                                      rmlo_output,
                                      ),
                                     axis=1
                                     )
            
            features.append(outputs)
            # features = np.append(features, outputs, axis=0)
            labels.append(targets.numpy())
            
    return np.concatenate(features), np.concatenate(labels)


class VGG_ONN(nn.Module):
    def __init__(self, architecture, in_channels=1, num_classes=3, q=3):
        super(VGG_ONN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.q = q
        self.features = self.create_conv_layers(architecture)

        self.fcs = nn.Sequential(
            nn.Linear(architecture[-2] * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for i in architecture:
            if type(i) == int:
                out_channels = i
                layers += [SelfONN2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                    q=self.q),
                        nn.BatchNorm2d(i),
                        nn.Tanh(),
                        ]
                in_channels = i
            elif i == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)