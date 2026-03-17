import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# -------------init-----------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(256 * 7 * 7, 2048) 
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 6) 
        
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)
#---------------------forward--------------------------
    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)

        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.adaptive_pool(x) 
        x = torch.flatten(x, 1)

        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)

# ----------------------------Training-------------------------------
def train_model(model, train_loader, val_loader, epochs, scheduler, optimizer, criterion, gpu_aug, device):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = gpu_aug(images)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
#---------------------------Validation----------------------------------------        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val)
                _, predicted = torch.max(preds.data, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved! With acc:{best_acc}")
    print(f"Finish Training, best acc {best_acc}")
#------------------------------------Testing--------------------------------------------------------
    #--------------------Testing-------------------
    model.eval()
    test_loss = 0
    correct_guess = 0 #because it's multiple class classfication 
    total_guess = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            test_loss += loss.item()

            _, predicted = torch.max(preds.data, 1) 
            total_guess += y_batch.size(0)
            correct_guess += (predicted == y_batch).sum().item()

    print(f"Final Loss: {test_loss/len(test_loader)}")
    print(f"Test Accuracy: {100 * correct_guess / total_guess:}%")
# -----------------main--------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_transform = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.0), antialias=True), 
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])

    gpu_aug = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    base_path = os.path.dirname(os.path.abspath(__file__))
    
    train_dataset = datasets.ImageFolder(os.path.join(base_path, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(base_path, 'test'), transform=gpu_aug)
    val_dataset = datasets.ImageFolder(os.path.join(base_path, 'val'), transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
    train_model(model, train_loader, val_loader, 60, scheduler, optimizer, criterion, gpu_aug, device)

    #----------------------------Confusion Matrix-------------------------------------

    def plot_cm(model, test_loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        
        classes = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title='Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.show()

    plot_cm(model, val_loader, device)

#Previous Milestone:
    """
#---------------------------Display 100 images-----------------------------

images, labels = next(iter(train_loader)) 
plt.figure(figsize=(15, 15))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray') 
    plt.title(train_dataset.classes[labels[i]], fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()

#----------------------loop dataloader--------------------------------
# ----------------------Test----------------------------
for inputs, outputs in test_loader:
    print("Test:\n")
    print(f"Input Data:\n{inputs}")
    print(f"Output Labels:\n{outputs}")
    break 

#------------------Val-------------------
for inputs, outputs in val_loader:
    print("Val:\n")
    print(f"Input Data:\n{inputs}")
    print(f"Output Labels:\n{outputs}")
    break

    
    """