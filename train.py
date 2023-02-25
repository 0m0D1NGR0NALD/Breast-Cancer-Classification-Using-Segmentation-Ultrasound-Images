import torch
from timeit import default_timer as timer
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy_fxn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    # How many of y_true == y_pred
    acc = (correct/len(y_pred))
    return acc

# Training and Testing Steps & Loop functions
def train_step(model,dataloader,loss_fxn,accuracy,optimizer,scheduler,device):
    model.train()
    train_loss, train_acc = 0,0
    for images,labels in dataloader:
        images,labels = images.to(device),labels.to(device)

        # Forward Pass
        logit = model(images)
        prob = torch.softmax(logit,dim=1)
        pred = torch.argmax(prob,dim=1)

        # Loss and Accuracy
        loss = loss_fxn(logit,labels)
        train_loss += loss
        train_acc += accuracy(y_true=labels,y_pred=pred)

        # Zero grad, back prop, step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()

    # Average loss and accuracy across all batches
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss,train_acc

def val_step(model,dataloader,loss_fxn,accuracy,device):
    model.eval()
    val_loss,val_acc = 0,0
    target_labels = []
    pred_probs = []

    # Turn off gradient tracking
    with torch.inference_mode():
        for images,labels in dataloader:
            images,labels = images.to(device),labels.to(device)

            # Forward Pass
            logit = model(images)
            prob = torch.softmax(logit,dim=1)
            pred = torch.argmax(prob,dim=1)

            pred_probs.append(prob.cpu())
            target_labels.append(labels.cpu())

            # Loss and Accuracy
            loss = loss_fxn(logit,labels)
            val_loss += loss
            val_acc += accuracy(y_true=labels,y_pred=pred)
        
        pred_probs = torch.cat(pred_probs)
        target_labels = torch.cat(target_labels)

        # Average loss and accuracy across all batches
        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

    return val_loss,val_acc

def train_model(epochs,model,train_loader,val_loader,loss_fxn,accuracy,optimizer,scheduler,device):
    # Create empty results dict that keeps track of metrics per epoch
    results = {
        'train_loss':[],
        'train_acc':[],
        'val_loss':[],
        'val_acc':[]
    }
    start_time = timer()

    for epoch in tqdm(range(epochs), desc='Training Model...'):
        train_loss,train_acc = train_step(model,train_loader,loss_fxn,accuracy,optimizer,scheduler,device)
        test_loss,test_acc = val_step(model,val_loader,loss_fxn,accuracy,device)

        print(f'Epoch: {epoch+1}\n----------')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {test_loss:.4f} | Val Acc: {test_acc*100:.2f}%')

        results['train_loss'].append(train_loss.item())
        results['train_acc'].append(train_acc)
        results['val_loss'].append(test_loss.item())
        results['val_acc'].append(test_acc)
    
    end_time = timer()
    print(f"Execution time on {device}: {format(end_time-start_time, '0.3f')} seconds.")

    return results,model

   