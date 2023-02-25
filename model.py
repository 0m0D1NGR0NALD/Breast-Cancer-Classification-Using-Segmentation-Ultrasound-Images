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
model = models.resnet18(weights=weights)
# Get number of input features into linear layer
features = model.fc.in_features
# Modify classifier output shape 
model.fc = nn.Linear(features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
accuracy = train.accuracy_fxn

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.225)

results,model = train.train_model(epochs,model,data.train_loader,data.val_loader,criterion,train.accuracy_fxn,optimizer,scheduler,device=device)
