import torch

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


