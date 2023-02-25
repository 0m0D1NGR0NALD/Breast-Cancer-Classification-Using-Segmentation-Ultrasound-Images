import matplotlib.pyplot as plt
import model

def plot_loss_acc(results):
    epochs = [i for i in range(len(results['val_acc']))]

    plt.figure(figsize=(20,6))

    plt.subplot(1,2,1)
    plt.plot(epochs,results['train_loss'],label='Train Loss Curve')
    plt.plot(epochs,results['val_loss'],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(epochs,results['train_acc'],label='Train Accuracy')
    plt.plot(epochs,results['val_acc'],label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')