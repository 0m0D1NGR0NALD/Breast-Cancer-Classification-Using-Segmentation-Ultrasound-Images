import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from pathlib import Path
from sklearn.model_selection import train_test_split
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = 'Dataset/'
path = Path(root_dir)

image_paths = list(path.glob('*/*.png'))

images = [str(image_path) for image_path in image_paths if '_mask' not in str(image_path)]
masks = [str(image_path) for image_path in image_paths if '_mask' in str(image_path)]
labels = [os.path.split(os.path.split(name)[0])[1] for name in images]

# Splitting data into train, test and validation set
train_data,test_data,train_labels,test_labels = train_test_split(images,labels,test_size=0.15,shuffle=True,random_state=12)
train_data,val_data,train_labels,val_labels = train_test_split(train_data,train_labels,test_size=0.15,shuffle=True,random_state=12)

# Masks
train_data_masks,test_data_masks,train_labels_masks,test_labels_masks = train_test_split(masks,labels,test_size=0.15,shuffle=True,random_state=12)
train_data_masks,val_data_masks,train_labels_masks,val_labels_masks = train_test_split(train_data_masks,train_labels_masks,test_size=0.15,shuffle=True,random_state=12)

# Transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )                                   
                                      ])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class Dataset(Dataset):
    def __init__(self,images:list,labels:list,transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transform(image)
        label = self.labels[index]
        return image, label


train_dataset = Dataset(images=train_data,labels=train_labels,transform=train_transforms)
val_dataset = Dataset(images=val_data,labels=val_labels,transform=val_transforms)
test_dataset = Dataset(images=test_data,labels=test_labels,transform=val_transforms)

train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=4,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False)