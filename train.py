import torch
import data

def accuracy(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    # How many of y_true == y_pred
    acc = (correct/len(y_pred))
    return acc

# Training and Testing Steps & Loop functions
def train_step(model,dataloader,loss_fn,accuracy,optimizer,scheduler,device):
    model.train()
    train_loss, train_acc = 0,0
    for batch_num, (images,labels) in enumerate(dataloader):
        images,labels = images.to(device),labels.to(device)

        # Forward Pass
        logit = model(images)
        prob = torch.softmax(logit,dim=1)
        pred = torch.argmax(prob,dim=1)

        # Loss and Accuracy
        loss = loss_fn(logit,labels)
        train_loss += loss
        train_acc += accuracy(y_true=labels,y_pred=pred)

        # Zero grad, back prop, step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Average loss and accuracy across all batches
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)

        return (train_loss,train_acc)


