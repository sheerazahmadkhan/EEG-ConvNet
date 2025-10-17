
import os
from pathlib import Path
import warnings
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Environment / Config
# -----------------------------------------------------------------------------
load_dotenv()

CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "data/labeledimagesspectrograms100.csv")
IMAGE_FOLDER_PATH = os.getenv("IMAGE_FOLDER_PATH", "data/testdata")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "2"))
NUM_MODELS = int(os.getenv("NUM_MODELS", "2"))
NUM_FOLDS = int(os.getenv("NUM_FOLDS", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "6"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "8e-05"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "models/saved_weights"))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel = str(self.df.iloc[idx, 0])
        img_name = os.path.join(self.root_dir, img_rel)
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image not found: {img_name}")
        image = Image.open(img_name).convert('RGB')
        # retain original +1 label shift (as in original script)
        label = int(self.df.iloc[idx, 1]) + 1

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 256 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def calculate_metrics(predictions, targets, losses):
    if len(predictions) == 0:
        return 0.0, 0.0, 0.0, 0.0, None, 0.0
    accuracy = (np.array(predictions) == np.array(targets)).mean() * 100.0
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(targets, predictions)
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    return accuracy, precision, recall, f1, cm, avg_loss


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# -----------------------------------------------------------------------------
# Training & evaluation utilities
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    train_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    train_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return train_loss, train_acc


def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    val_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return val_loss, val_acc, predictions, targets


# --------------------------------------------------------------------------------------------
# Main training script 
# ---------------------------------------------------------------------------------------------
def main():
    # Prepare dataset
    csv_file_path = CSV_FILE_PATH
    image_folder_path = IMAGE_FOLDER_PATH

    print("Loading dataset...")
    dataset = CustomDataset(csv_file_path, image_folder_path, transform=transform)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    all_best_model_weights = []  # list of tuples: (fold, model_idx, path)

    for fold, (train_index, val_index) in enumerate(skf.split(dataset.df, dataset.df.iloc[:, 1])):
        print(f"\n=== Fold {fold+1}/{NUM_FOLDS} ===")

        # create fold datasets / dataloaders
        train_df = dataset.df.iloc[train_index].reset_index(drop=True)
        val_df = dataset.df.iloc[val_index].reset_index(drop=True)
        test_df = dataset.df[~dataset.df.index.isin(train_index)].reset_index(drop=True)

        train_dataset = CustomDataset(csv_file_path, image_folder_path, transform=transform)
        val_dataset = CustomDataset(csv_file_path, image_folder_path, transform=transform)
        test_dataset = CustomDataset(csv_file_path, image_folder_path, transform=transform)

        train_dataset.df = train_df
        val_dataset.df = val_df
        test_dataset.df = test_df

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        models_list = []
        val_predictions_per_model = []

        for model_idx in range(NUM_MODELS):
            print(f"\nTraining model {model_idx+1}/{NUM_MODELS} for fold {fold+1}")

            model = CustomCNN(num_classes=3)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            best_val_loss = float('inf')
            patience = 3
            early_stopping_counter = 0
            best_state = None

            for epoch in range(NUM_EPOCHS):
                start_time = time.time()
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc, val_preds, val_targets = eval_model(model, val_loader, criterion, device)

                print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | Time: {time.time()-start_time:.1f}s")

                scheduler.step()

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    best_state = model.state_dict()
                    save_path = WEIGHTS_DIR / f"model_fold{fold+1}_model{model_idx+1}_best.pth"
                    torch.save(best_state, save_path)
                    print(f"Saved best model to: {save_path}")
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} for model {model_idx+1} (no improvement for {patience} epochs).")
                    break

            if best_state is not None:
                models_list.append(best_state)
                all_best_model_weights.append((fold+1, model_idx+1, str(WEIGHTS_DIR / f"model_fold{fold+1}_model{model_idx+1}_best.pth")))
            else:
                print(f"Warning: no best state saved for fold {fold+1} model {model_idx+1}")

            
            try:
                if len(val_preds) > 0 and len(val_targets) > 0:
                    cm = confusion_matrix(val_targets, val_preds)
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d")
                    plt.title(f"Confusion Matrix - Fold{fold+1}_Model{model_idx+1}")
                    plt.show()
            except Exception as e:
                print("Could not plot conf matrix:", e)

            # Evaluate on test set for this model
            test_preds = []
            with torch.no_grad():
                temp_model = CustomCNN(num_classes=3).to(device)
                temp_model.load_state_dict(best_state if best_state is not None else model.state_dict())
                temp_model.eval()
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = temp_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_preds.extend(predicted.cpu().numpy())
            val_predictions_per_model.append(test_preds)

        # If no models produced predictions, skip ensemble evaluation
        if len(val_predictions_per_model) == 0:
            print(f"No model predictions for fold {fold+1}; skipping ensemble for this fold.")
            continue

        # Ensemble (average predictions then round down to int)
        ensemble_preds = np.mean(np.array(val_predictions_per_model), axis=0).astype(int)
        test_targets = test_dataset.df.iloc[:, 1].values if test_dataset.df.shape[0] > 0 else []

        accuracy, precision, recall, f1, cm, avg_loss = calculate_metrics(ensemble_preds, test_targets, [])
        print(f"Ensemble Metrics (Fold {fold+1}) -> Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if cm is not None:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title(f"Ensemble Confusion Matrix - Fold {fold+1}")
            plt.show()

    # Save ensemble collection
    if len(all_best_model_weights) > 0:
        ensemble_save = {"models": []}
        for (fidx, midx, pth) in all_best_model_weights:
            try:
                state = torch.load(pth, map_location='cpu')
                ensemble_save["models"].append({"fold": fidx, "model_idx": midx, "state_dict": state})
            except Exception as e:
                print(f"Could not load {pth}: {e}")

        ensemble_path = WEIGHTS_DIR / "ensemble_best.pth"
        torch.save(ensemble_save, ensemble_path)
        print(f"Saved ensemble collection to: {ensemble_path}")
    else:
        print("No best model weights found; ensemble file not created.")


if __name__ == "__main__":
    main()
