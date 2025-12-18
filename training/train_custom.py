import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- CUSTOM TUNING CONFIGURATION ---
DATA_DIR = "../Friends_Dataset_Split"  # The folder with your 80/20 split
WEIGHTS_PATH = "../fer_final_v2.pth"   # The 69.66% FER model
BATCH_SIZE = 32                     # Smaller batch size for better generalization on small data
LEARNING_RATE = 1e-4                # Low LR to gently fine-tune
EPOCHS = 15                         # Short run (Model learns specific faces fast)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------

def get_data_loaders():
    """
    Pipeline: Resizes variable friend images to 224x224.
    Uses ImageNet Normalization (Critical for the pre-trained weights).
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            
            # Robust Augmentation for Webcam Variance
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            
            transforms.ToTensor(),
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

def load_transfer_model(num_classes):
    """
    Loads ResNet18 and injects the FER-2013 weights.
    """
    print(f"🔧 Initializing ResNet18 Skeleton...")
    model = models.resnet18(weights=None) # We load our OWN weights, not ImageNet
    
    # 1. Match architecture to the saved file (Input: 3 channels, Output: 7 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 2. Load the FER-2013 Brain
    if os.path.exists(WEIGHTS_PATH):
        print(f"💉 Injecting Weights from '{WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("✅ Weights Loaded Successfully. The model knows what a face is.")
    else:
        print(f"❌ CRITICAL ERROR: '{WEIGHTS_PATH}' not found!")
        print("   Please make sure the FER model is in the same folder.")
        exit()
        
    return model.to(DEVICE)

def train_custom():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: '{DATA_DIR}' not found. Run create_split.py first.")
        return

    dataloaders, dataset_sizes, class_names = get_data_loaders()
    print(f"✅ Classes: {class_names}")
    print(f"🚀 Starting Fine-Tuning on {DEVICE} for {EPOCHS} epochs...")

    model = load_transfer_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam with Low LR
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
                    torch.save(model.state_dict(), 'custom_final.pth')
                    print(f"💾 New Best Model Saved! Accuracy: {best_acc:.4f}")

    print(f'\n🏆 Final Personalized Accuracy: {best_acc:.4f}')
    print("✨ Model saved as 'custom_final.pth'")

if __name__ == "__main__":
    train_custom()
