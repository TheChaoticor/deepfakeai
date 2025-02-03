import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from main import DeepFakeDetector
from torch.optim.lr_scheduler import StepLR
import numpy as np

class DeepFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images
        for img_name in os.listdir(real_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(real_dir, img_name))
                self.labels.append(0)  # 0 for real
                
        # Load fake images
        for img_name in os.listdir(fake_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(fake_dir, img_name))
                self.labels.append(1)  # 1 for fake
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss()
    
    best_val_loss = np.inf
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).squeeze()
                val_loss += criterion(outputs, labels.float()).item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {100*correct/total:.2f}%')
        print('--------------------')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'weights/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
    
    return model

def main():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DeepFakeDataset(
        real_dir='data/real',
        fake_dir='data/fake',
        transform=transform
    )
    
    val_dataset = DeepFakeDataset(
        real_dir='data/real',
        fake_dir='data/fake',
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = DeepFakeDetector()
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler)
    
    # Save model
    torch.save(trained_model.state_dict(), 'weights/model.pth')

if __name__ == "__main__":
    main()