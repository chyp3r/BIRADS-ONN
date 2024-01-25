import numpy as np
import albumentations as A

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from albumentations.pytorch import ToTensorV2


class Transforms:
    """
    Dataloader for train cases
    """
    def __init__(self):
        # Data augmentation methods
        self.transforms = A.Compose([
        A.CLAHE(),
        A.RandomGamma(),
        A.GridDistortion(num_steps=12,),
        A.ElasticTransform(),
        A.HorizontalFlip(),
        A.SafeRotate(limit=45),
        A.Normalize(),
        ToTensorV2(),
    ])

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

class TestTransforms:
    """
    Dataloader for test cases
    """
    def __init__(self):
        # Data augmentation methods
        self.transforms = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])  

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']       


train_dataset = datasets.ImageFolder("BI-RADS_DATASET_SPLIT0\\train",transform=Transforms())  # Change path
test_dataset = datasets.ImageFolder("BI-RADS_DATASET_SPLIT0\\val",transform=TestTransforms()) # Change path
try_set = datasets.ImageFolder("Denemeset",transform=TestTransforms()) # Change path

train_loader_f = DataLoader(train_dataset,48,sampler=ImbalancedDatasetSampler(train_dataset))
test_loader_f = DataLoader(test_dataset,48 )
try_set = DataLoader(try_set,1)



