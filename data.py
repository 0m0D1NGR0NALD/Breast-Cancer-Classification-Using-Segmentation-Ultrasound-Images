import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from pathlib import Path
import os

root_dir = 'Dataset/'
path = Path(root_dir)

image_paths = list(path.glob('*/*.png'))

images = [str(image_path) for image_path in image_paths if '_mask' not in str(image_path)]
labels = [os.path.split(os.path.split(name)[0])[1] for name in images]

print([os.path.split(name)[0] for name in images])