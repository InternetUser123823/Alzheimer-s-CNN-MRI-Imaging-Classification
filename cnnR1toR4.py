from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim

IMG_SIZE = (208, 176)
BATCH_SIZE = 32

# Train transform: augmentation + normalization
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),          # scales pixels to [0, 1] automatically
])

# Val/test transform: no augmentation, just resize + normalize
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ImageFolder infers labels from subfolder names, same as Keras's image_dataset_from_directory
full_train = datasets.ImageFolder("original_data/train", transform=train_transform)
# Need a non-augmented training set for a loader for consistent F1 score comparison
full_train_eval = datasets.ImageFolder("original_data/train", transform=eval_transform)
test_ds = datasets.ImageFolder("original_data/test", transform=eval_transform)

print("Classes:", full_train.classes)  # alphabetical, same indexing caveat as before

# Split train into train/val (80/20), reproducibly
# All labels for the full training folder (not just the current split)
all_labels = full_train.targets  # list, one label per image in original_data/train
all_indices = list(range(len(full_train)))

train_idx, val_idx = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=all_labels,   
    random_state=42,
)

train_ds = torch.utils.data.Subset(full_train, train_idx)
train_eval_ds = torch.utils.data.Subset(full_train_eval, train_idx)      

# Rebuild train_labels for this split, needed for both class_weights (if still used)
# and the WeightedRandomSampler
train_labels = [full_train.targets[i] for i in train_idx]

# Prevent validation set from being augmented
full_train_eval = datasets.ImageFolder("original_data/train", transform=eval_transform)
val_ds = torch.utils.data.Subset(full_train_eval, val_idx) 

# # Ensure everything is working right
# images, labels = next(iter(train_loader))
# print("Batch shape:", images.shape)   
# print("Pixel range:", images.min().item(), "-", images.max().item())  

# Class weights, from the actual training set
train_labels = [full_train.targets[i] for i in train_ds.indices]

# Remove class weights because biased sampling is already in place
# weights = compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(train_labels),
#     y=train_labels,
# )

# Incorporate WeightedRandomSampler such that every batch includes a roughly equal mix of all 4 classes
class_sample_counts = np.array([np.sum(np.array(train_labels) == c) for c in np.unique(train_labels)])
sample_weights = 1.0 / np.sqrt(class_sample_counts[train_labels])
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# # Softened class weight (sqrt-based instead of full inverse) to ensure rare classes still cost more
# class_counts_for_weight = np.array([np.sum(np.array(train_labels) == c) for c in np.unique(train_labels)])
# loss_weights = 1.0 / np.sqrt(class_counts_for_weight)
# loss_weights = loss_weights / loss_weights.sum() * len(loss_weights)  # normalize the new weights
# class_weights = torch.tensor(loss_weights, dtype=torch.float32)

# Loaders
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Need a non-augmented, non-sampled training loader as well for consistent F1 score comparison
train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False)
# New augmented loader that includes specialized sampler
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

# Remove class weights because biased sampling is already in place
# class_weights = torch.tensor(weights, dtype=torch.float32)
# print("Class weights:", class_weights)

# Create CNN 
class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96), # normalize activations within each batch
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2), 
        )

        # Updated AdaptiveAvgPool 
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Collapses each feature map into one number
            nn.Flatten(),               
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Use GPU acceleration if available (MPS = Apple Silicon GPU, CUDA = Nvidia)
device = torch.device("mps" if torch.backends.mps.is_available() 
                       else "cuda" if torch.cuda.is_available() 
                       else "cpu")
print("Using device:", device)

model = AlzheimerCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001,  weight_decay=1e-4) # 10x lower to avoid weight "lurching"
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

NUM_EPOCHS = 50

# Train model in one epoch
def train_one_epoch():
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from last batch
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # caps gradients if they are too large
        optimizer.step()               # update weights

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# Calculates loss and accuracy of model after epoch 
def evaluate(loader):
    model.eval()  # disables dropout
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradients needed, saves memory/compute
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# Evaluates F1 score after epoch
def evaluate_with_f1(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro')

# Run the model and evaluate it
best_val_f1 = 0.0

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate(val_loader)
    train_macro_f1 = evaluate_with_f1(train_eval_loader)
    val_macro_f1 = evaluate_with_f1(val_loader)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_macro_f1:.4f} | "
          f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_macro_f1:.4f}")

    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  -> Saved new best model (val_f1: {val_macro_f1:.4f})")

    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     torch.save(model.state_dict(), "best_model.pt")
    #     print(f"  -> Saved new best model (val_acc: {val_acc:.4f})")

# Print out results for best model

class_names = full_train.classes  # ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def evaluate_detailed(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(all_labels, all_preds))

# Load the BEST checkpoint, not whatever the model's weights happen to be
# after the last epoch (epoch 20's weights are probably not the best ones,
# given val was crashing right up to the end)
best_model = AlzheimerCNN(num_classes=4).to(device)
best_model.load_state_dict(torch.load("best_model.pt"))

print("\n=== Best model, evaluated on validation set ===")
evaluate_detailed(best_model, val_loader)