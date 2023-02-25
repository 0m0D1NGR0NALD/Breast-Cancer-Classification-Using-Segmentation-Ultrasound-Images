import torch
from torchmetrics import Precision,Recall,F1Score
from sklearn.metrics import roc_auc_score
import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(model,dataloader,device):
    # Storing pred labels
    prediction_labels = []
    prediction_probs = []

    # Testing all testing data
    model.eval()

    with torch.inference_mode():
        for images, labels in dataloader:
            images,labels = images.to(device), labels.to(device)

            logits = model(images)
            prob = torch.softmax(logits, dim=1)
            labels = torch.argmax(prob, dim=1)

            prediction_labels.append(labels.cpu())
            prediction_probs.append(prob.cpu())

        # Concatenate all tensors in predictions list into big tensor
        prediction_labels = torch.cat(prediction_labels)
        prediction_probs = torch.cat(prediction_probs)

        return prediction_labels,prediction_probs

target_labels = []
for img,label in data.test_dataset:
    target_labels.append(label)

accuracy = Accuracy().to(device)
precision = Precision().to(device)
recall = Recall().to(device)
f1_score = F1Score(num_classes=4)

predicted_labels,predicted_probs = test_model(model,data.test_loader,device=device)

test_accuracy = accuracy(predicted_labels, torch.Tensor(target_labels).type(dtype=torch.int))
test_precision = precision(predicted_labels, torch.Tensor(target_labels).type(dtype=torch.int))
test_recall = recall(predicted_labels, torch.Tensor(target_labels).type(dtype=torch.int))
f1_score = f1_score(predicted_labels, torch.Tesor(target_labels).type(dtype=torch.int))
auc = roc_auc_score(target_labels, predicted_probs, multi_class='ovr')

print(f"Test Accuracy: {test_accuracy.item():.4f}")
print(f"Test Precision: {test_precision.item():.4f}")
print(f"Test Recall: {test_recall.item():.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"AUC: {auc:.4f}")
