import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- FINAL CONFIGURATION ---
DATA_DIR = "fer2013"
BATCH_SIZE = 64
LEARNING_RATE = 1e-4        # Low, constant LR (Adam)
EPOCHS = 30                 # Sufficient for fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------

def get_data_loaders():
    """
    Fixed Pipeline: Uses ImageNet Normalization and Stronger Augmentation
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # Convert to 3 channels for ResNet
            transforms.Resize((224, 224)),
            
            # Augmentation from your notebook + slight boost
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Added Affine!
            
            transforms.ToTensor(),
            # CRITICAL FIX: Use ImageNet Normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['train', 'test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=4)
        for x in ['train', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

def get_final_model(num_classes):
    """
    ResNet-18 Standard (No complex head)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # We do NOT hack the input conv1 this time. 
    # We rely on Grayscale(3) transform to feed it 3 channels as it expects.
    
    # Simple Head (As per your notebook)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model.to(DEVICE)

def train_model():
    dataloaders, dataset_sizes, class_names = get_data_loaders()
    print(f"✅ Classes: {class_names}")
    print(f"🚀 Final Training (Adam 1e-4 + ImageNet Norm) on {DEVICE}...")

    model = get_final_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    
    # OPTIMIZER: Adam with Constant LR (No Decay)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'fer_final_v2.pth')
                    print("💾 New Best Model Saved!")

    print(f'\n🏆 Best Validation Accuracy: {best_acc:.4f}')

if __name__ == "__main__":
    train_model()
