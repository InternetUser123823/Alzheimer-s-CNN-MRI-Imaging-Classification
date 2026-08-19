from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler, DataLoader, Subset
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim

IMG_SIZE = (208, 176)
BATCH_SIZE = 32
N_SPLITS = 5

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
            nn.AdaptiveMaxPool2d(output_size=(34, 29)), 

            # Extra Block
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
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

# Train model in one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, device):
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
def evaluate(model, loader, criterion, device):
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
def evaluate_with_f1(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)

# Implement focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Use GPU acceleration if available (MPS = Apple Silicon GPU, CUDA = Nvidia)
device = torch.device("mps" if torch.backends.mps.is_available() 
                       else "cuda" if torch.cuda.is_available() 
                       else "cpu")
print("Using device:", device)

# Stratified K-Fold
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

all_labels = np.array(full_train.targets)  # list, one label per image in original_data/train
# all_indices = list(range(len(full_train)))

fold_results = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(np.zeros(len(all_labels)), all_labels)):
    print(f"\n{'='*20} FOLD {fold + 1}/{N_SPLITS} {'='*20}")

    train_ds = torch.utils.data.Subset(full_train, train_idx)
    # Non-augmented version of the training set (see loaders for more detail)
    train_eval_ds = torch.utils.data.Subset(full_train_eval, train_idx)      

    # Rebuild train_labels for this split, needed for the WeightedRandomSampler
    train_labels = all_labels[train_idx]

    # Prevent validation set from being augmented
    val_ds = torch.utils.data.Subset(full_train_eval, val_idx) 

    # # Ensure everything is working right
    # images, labels = next(iter(train_loader))
    # print("Batch shape:", images.shape)   
    # print("Pixel range:", images.min().item(), "-", images.max().item())  

    # (Removed) class weights because biased sampling is already in place
    # weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=np.unique(train_labels),
    #     y=train_labels,
    # )

    # Incorporate WeightedRandomSampler such that every batch includes a roughly equal probability of each class being selected
    class_sample_counts = np.bincount(train_labels)
    sample_weights = 1.0 / np.sqrt(class_sample_counts[train_labels])
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # # (Removed) softened class weight (sqrt-based instead of full inverse) to ensure rare classes still cost more
    # class_counts_for_weight = np.array([np.sum(np.array(train_labels) == c) for c in np.unique(train_labels)])
    # loss_weights = 1.0 / np.sqrt(class_counts_for_weight)
    # loss_weights = loss_weights / loss_weights.sum() * len(loss_weights)  # normalize the new weights
    # class_weights = torch.tensor(loss_weights, dtype=torch.float32)

    # Loaders
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Need a non-augmented, non-sampled training loader as well for consistent F1 score comparison
    train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False)
    # New augmented loader that includes specialized sampler
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

    # Remove class weights because biased sampling is already in place
    # class_weights = torch.tensor(weights, dtype=torch.float32)
    # print("Class weights:", class_weights)

    # Create model
    model = AlzheimerCNN(num_classes=4).to(device)
    criterion = FocalLoss(gamma=1.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001,  weight_decay=1e-4) # 10x lower to avoid weight "lurching"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    NUM_EPOCHS = 50

    # Train the model and evaluate it
    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None
    patience = 7          # how many epochs to tolerate with no improvement
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_macro_f1 = evaluate_with_f1(model, train_eval_loader)
        val_macro_f1 = evaluate_with_f1(model, val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_macro_f1:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            epochs_without_improvement = 0
            best_epoch = epoch + 1

            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> Saved new best model (val_f1: {val_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping — no val F1 improvement in {patience} epochs (best: {best_val_f1:.4f})")
                break

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), "best_model.pt")
        #     print(f"  -> Saved new best model (val_acc: {val_acc:.4f})")

    fold_results.append({"fold": fold + 1, "best_state": best_state, "best_val_f1": best_val_f1, "val_loader": val_loader, "best_epoch": best_epoch})

class_names = full_train.classes  # ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Print out all fold metrics
fold_f1s = [result["best_val_f1"] for result in fold_results]

print("\n=== 5-Fold Cross-Validation Results ===")
for result in fold_results:
    print(
        f"Fold {result['fold']}: "
        f"Best macro F1 = {result['best_val_f1']:.4f}"
    )

print(
    f"\nMacro F1: {np.mean(fold_f1s):.4f} "
    f"± {np.std(fold_f1s, ddof=1):.4f}"
)

# Print out results for best model

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

    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(all_labels, all_preds))

# Load the BEST checkpoint, not whatever the model's weights happen to be
# after the last epoch (epoch 20's weights are probably not the best ones,
# given val was crashing right up to the end)
best_fold_dict = max(fold_results, key=lambda x: x["best_val_f1"])
best_model_state = best_fold_dict["best_state"]
best_val_loader = best_fold_dict["val_loader"]

best_model = AlzheimerCNN(num_classes=4).to(device)
best_model.load_state_dict(best_model_state)

print("\n=== Best model, evaluated on validation set ===")
evaluate_detailed(best_model, best_val_loader)

# Evaluate based on test dataset
# Create model
final_model = AlzheimerCNN(num_classes=4).to(device)
final_criterion = FocalLoss(gamma=1.5)
final_optimizer = optim.Adam(final_model.parameters(), lr=0.0001,  weight_decay=1e-4) # 10x lower to avoid weight "lurching"

# Create full training loader using ALL training labels
final_train_labels = np.array(full_train.targets)
final_class_counts = np.bincount(final_train_labels)

# Weighted sampler
final_sample_weights = 1.0 / np.sqrt(
    final_class_counts[final_train_labels]
)

final_sample_weights = torch.tensor(
    final_sample_weights,
    dtype=torch.float32
)

final_sampler = WeightedRandomSampler(
    final_sample_weights,
    num_samples=len(final_sample_weights),
    replacement=True
)

final_train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, sampler=final_sampler)
final_train_eval_loader = DataLoader(full_train_eval, batch_size=BATCH_SIZE, shuffle=False)
final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=3)

final_num_epochs = int(np.median(
    [result["best_epoch"] for result in fold_results]
))
for epoch in range(final_num_epochs):
        train_loss, train_acc = train_one_epoch(final_model, final_train_loader, final_criterion, final_optimizer, device)
        train_macro_f1 = evaluate_with_f1(final_model, final_train_eval_loader)

        final_scheduler.step(train_loss)
        
        print(f"Epoch {epoch+1}/{final_num_epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_macro_f1:.4f} | ")

print("\n=== Final model, evaluated on test set ===")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False) 
evaluate_detailed(final_model, test_loader)
