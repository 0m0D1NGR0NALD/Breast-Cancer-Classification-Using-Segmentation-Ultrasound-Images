import torch
from torchvision import models
import torch.nn as nn
import train
import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 224
num_classes = 3
epochs = 2

weights = models.ResNet18_Weights.DEFAULT
# Instatiate
model_ft = models.resnet18(weights=weights)
# Get number of input features into linear layer
num_ftrs = model_ft.fc.in_features
# Modify classifier output shape 
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
accuracy = train.accuracy_fn

optimizer = torch.optim.Adam(model_ft.parameters(),lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.225)

results,model = train.train_model(epochs,model_ft,data.train_loader,data.val_loader,criterion,train.accuracy_fn,optimizer,scheduler,device=device)
