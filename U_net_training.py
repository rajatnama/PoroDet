#Unet architechure, training, metrics and plotting 
# import the required libraries, pytorch for model and training

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import KFold

# Performance matrics of model training
try:
    from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc, f1_score, jaccard_score, precision_score,
        recall_score, confusion_matrix)
except Exception as e:
    raise ImportError("matics not found. Install it with  pip install scikit-learn`.") from e

# UNet model architeccture, used double conv 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.2):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512, dropout_rate)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024, dropout_rate)

        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512, dropout_rate)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256, dropout_rate)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128, dropout_rate)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64, dropout_rate)

        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)
        return conv10

# define dataset class. Group files by original images and their augmentations
def get_original_and_augmented_groups(data_dir):

    # consider .tif images only
    all_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.tif')])
    original_groups = {}
    for file in all_files:
        if "_aug_" in file:
            original_name = file.split("_aug_")[0]
            try:
                aug_num = int(file.split("_aug_")[1].split('.')[0])
            except Exception:
                aug_num = 0
            original_groups.setdefault(original_name, []).append((file, aug_num))
        else:
            # also include originals even if no augmentations
            original_groups.setdefault(file.replace('.tif',''), [])

    # sort augmentation lists
    for k in list(original_groups.keys()):
        original_groups[k] = sorted(original_groups[k], key=lambda x: x[1])
    return original_groups

# Define how dataset will be loaded during training and validation
class Nanopore_Dataset(Dataset):
    def __init__(self, data_dir, original_images, is_validation=False, target_size=(1024,1024)): #Change target size as per need
        self.data_dir = data_dir
        self.is_validation = is_validation
        self.target_size = target_size

        all_files = sorted(os.listdir(data_dir))
        image_files = []
        for file in all_files:
            if file.lower().endswith('.tif'):
                original_name = file.split("_aug_")[0] if "_aug_" in file else file.replace('.tif','')
                if original_name in original_images:
                    mask_file = file.replace('.tif', '_mask.png')
                    if mask_file in all_files:
                        image_files.append(file)
        self.image_files = image_files
        print(f"{'Validation' if is_validation else 'Training'} set contains {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.data_dir, img_name)
            mask_path = os.path.join(self.data_dir, img_name.replace('.tif', '_mask.png'))

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                raise ValueError(f"Could not read {img_name} or mask")

            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

            image = image.astype(np.float32) / 255.0
            mask = (mask.astype(np.float32) / 255.0)  

            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)

            return image, mask
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            placeholder = torch.zeros((1, 1024, 1024), dtype=torch.float32)
            return placeholder, placeholder

# Dfine performance metrics functions
def dice_coeff(preds_bool, targets_bool, eps=1e-7):
    preds = preds_bool.astype(np.uint8)
    targets = targets_bool.astype(np.uint8)
    intersection = (preds & targets).sum()
    return (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)

def iou_score_np(preds_bool, targets_bool, eps=1e-7):
    preds = preds_bool.astype(np.uint8)
    targets = targets_bool.astype(np.uint8)
    intersection = (preds & targets).sum()
    union = (preds | targets).sum()
    return (intersection + eps) / (union + eps)

def check_overfitting(train_loss, val_loss, threshold=0.3):
    if val_loss > 0 and train_loss < val_loss * (1 - threshold):
        return True
    return False

# Modal training 
def train_nanopore_detector(data_dir, output_dir, train_originals, val_originals, batch_size=1, epochs=30, learning_rate=5e-5, 
                            patience=3, weight_decay=1e-4, fold_id=1, prob_threshold=0.5, 
                            resize_to=(1024,1024)): #Change the training parameters as per need for better training
    
    #use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nFold {fold_id}: Using device {device}")
    print(f"Fold {fold_id}: {len(train_originals)} originals for training, {len(val_originals)} for validation")
    print(f"Fold {fold_id}: Validation originals: {val_originals}")

    train_dataset = Nanopore_Dataset(data_dir, train_originals, is_validation=False, target_size=resize_to)
    val_dataset = Nanopore_Dataset(data_dir, val_originals, is_validation=True, target_size=resize_to)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True 
                              if torch.cuda.is_available() else False)
    val_loader = DataLoader( val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True 
                            if torch.cuda.is_available() else False)

    model = UNet(in_channels=1, out_channels=1, dropout_rate=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    early_stop_counter = 0

    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
             'val_iou': [], 'val_dice': [], 'val_pr_auc': [], 'val_roc_auc': []}

    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    #Display training parameters and details
    print("\nStarting training with verbose metrics to monitor progress")
    print(f"Early stopping patience: {patience}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")

    try:
        for epoch in range(epochs):
            # trainig set
            model.train()
            running_train_loss = 0.0
            train_probs_list = []
            train_targets_list = []

            with tqdm(train_loader, desc=f"Fold {fold_id} | Epoch {epoch+1}/{epochs} [train]") as pbar:
                for images, masks in pbar:
                    images, masks = images.to(device), masks.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()
                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    targets = masks.detach().cpu().numpy()
                    train_probs_list.append(probs.reshape(-1))
                    train_targets_list.append(targets.reshape(-1))

                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            avg_train_loss = running_train_loss / max(1, len(train_loader))
            history['train_loss'].append(avg_train_loss)

            if len(train_probs_list) > 0 and len(train_targets_list) > 0:
                all_train_probs = np.concatenate(train_probs_list)
                all_train_targets = np.concatenate(train_targets_list)
                train_bin_preds = (all_train_probs >= prob_threshold).astype(np.uint8)
                train_bin_targets = (all_train_targets >= 0.5).astype(np.uint8)
                train_accuracy = (train_bin_preds == train_bin_targets).mean()
            else:
                train_accuracy = float('nan')

            history['train_accuracy'].append(float(train_accuracy))

            # Validation set
            model.eval()
            running_val_loss = 0.0
            val_probs_list = []
            val_targets_list = []
            vis_saved = False

            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(val_loader):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    running_val_loss += loss.item()

                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    targets = masks.detach().cpu().numpy()
                    val_probs_list.append(probs.reshape(-1))
                    val_targets_list.append(targets.reshape(-1))

                    if (epoch % 5 == 0) and (not vis_saved):
                        for i in range(min(2, images.size(0))):
                            img_np = images[i].detach().cpu().numpy()[0]
                            gt_np = targets[i][0]
                            pred_np = probs[i][0]
                            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                            axes[0].imshow(img_np, cmap='gray'); axes[0].set_title('TEM'); axes[0].axis('off')
                            axes[1].imshow(gt_np, cmap='gray'); axes[1].set_title('Mask'); axes[1].axis('off')
                            axes[2].imshow(pred_np, cmap='gray'); axes[2].set_title('Pred Prob'); axes[2].axis('off')
                            plt.tight_layout()
                            plt.savefig(os.path.join(vis_dir, f'fold_{fold_id}_epoch_{epoch}_val_sample_{i}.png'))
                            plt.close()
                        vis_saved = True

            avg_val_loss = running_val_loss / max(1, len(val_loader))
            history['val_loss'].append(avg_val_loss)

            if len(val_probs_list) > 0 and len(val_targets_list) > 0:
                all_val_probs = np.concatenate(val_probs_list)
                all_val_targets = np.concatenate(val_targets_list)
                val_bin_preds = (all_val_probs >= prob_threshold).astype(np.uint8)
                val_bin_targets = (all_val_targets >= 0.5).astype(np.uint8)

                val_accuracy = (val_bin_preds == val_bin_targets).mean()
                prec = precision_score(val_bin_targets, val_bin_preds, zero_division=0)
                rec = recall_score(val_bin_targets, val_bin_preds, zero_division=0)
                f1 = f1_score(val_bin_targets, val_bin_preds, zero_division=0)
                try:
                    iou = jaccard_score(val_bin_targets, val_bin_preds, zero_division=0)
                except Exception:
                    iou = iou_score_np(val_bin_preds.astype(bool), val_bin_targets.astype(bool))
                dice = dice_coeff(val_bin_preds.astype(bool), val_bin_targets.astype(bool))
                try:
                    pr_auc = average_precision_score(val_bin_targets, all_val_probs)
                except Exception:
                    pr_auc = float('nan')
                has_pos = val_bin_targets.sum() > 0
                has_neg = (val_bin_targets.size - val_bin_targets.sum()) > 0
                try:
                    roc_auc = roc_auc_score(val_bin_targets, all_val_probs) if (has_pos and has_neg) else float('nan')
                except Exception:
                    roc_auc = float('nan')
            else:
                val_accuracy = prec = rec = f1 = iou = dice = pr_auc = roc_auc = float('nan')
                all_val_probs = np.array([])
                all_val_targets = np.array([])

            history['val_accuracy'].append(float(val_accuracy))
            history['val_precision'].append(float(prec))
            history['val_recall'].append(float(rec))
            history['val_f1'].append(float(f1))
            history['val_iou'].append(float(iou))
            history['val_dice'].append(float(dice))
            history['val_pr_auc'].append(float(pr_auc) if not np.isnan(pr_auc) else None)
            history['val_roc_auc'].append(float(roc_auc) if not np.isnan(roc_auc) else None)

            print(
                f"Fold {fold_id} & Epoch {epoch+1}: "
                f"Train Loss {avg_train_loss:.4f}, Train Acc {train_accuracy:.4f} | "
                f"Val Loss {avg_val_loss:.4f}, Val Acc {val_accuracy:.4f} | "
                f"Val Prec {prec:.4f}, Rec {rec:.4f}, F1 {f1:.4f}, IoU {iou:.4f}"
                f"PR-AUC {pr_auc if not np.isnan(pr_auc) else 'NA'}, ROC-AUC {roc_auc if not np.isnan(roc_auc) else 'NA'}" )

            if check_overfitting(avg_train_loss, avg_val_loss):
                print(" WARNING: Potential overfitting detected. Stop model training and try adjusting hyperparameters.")

            # checkpoints 
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': avg_train_loss,
                        'train_accuracy': train_accuracy, 'val_loss': avg_val_loss, 'val_accuracy': val_accuracy}, 
                        os.path.join(output_dir, f'fold_{fold_id}_checkpoint_epoch_{epoch}.pth'))

            # best model for this fold
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy, 'val_loss': avg_val_loss, 'val_accuracy': val_accuracy}, 
                    os.path.join(output_dir, f'fold_{fold_id}_best_model.pth'))
                print(f"Fold {fold_id}: Saved new best model (val loss {best_val_loss:.4f})")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"Fold {fold_id}: Validation loss did not improve. Counter: {early_stop_counter}/{patience}")
                if early_stop_counter >= patience:
                    print(f"Fold {fold_id}: Early stopping triggered after {epoch+1} epochs")
                    break

            scheduler.step(avg_val_loss)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # final model save
        final_model_path = os.path.join(output_dir, f'fold_{fold_id}_final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Fold {fold_id}: Training finished. Final model saved to: {final_model_path}")

        # metrics CSV for this fold
        csv_path = os.path.join(output_dir, f'fold_{fold_id}_metrics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [ 'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_iou', 'val_dice',
                       'val_pr_auc', 'val_roc_auc' ]
            writer.writerow(header)
            for i in range(len(history['train_loss'])):
                row = [
                    i + 1,
                    history['train_loss'][i],
                    history['train_accuracy'][i] if i < len(history['train_accuracy']) else '',
                    history['val_loss'][i] if i < len(history['val_loss']) else '',
                    history['val_accuracy'][i] if i < len(history['val_accuracy']) else '',
                    history['val_precision'][i] if i < len(history['val_precision']) else '',
                    history['val_recall'][i] if i < len(history['val_recall']) else '',
                    history['val_f1'][i] if i < len(history['val_f1']) else '',
                    history['val_iou'][i] if i < len(history['val_iou']) else '',
                    history['val_dice'][i] if i < len(history['val_dice']) else '',
                    history['val_pr_auc'][i] if i < len(history['val_pr_auc']) else '',
                    history['val_roc_auc'][i] if i < len(history['val_roc_auc']) else ''
                ]
                writer.writerow(row)
        print(f"Fold {fold_id}: Saved metrics CSV to: {csv_path}")

        # loss curves
        plt.figure(figsize=(8, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.title(f'Loss curves (Fold {fold_id})')
        plt.savefig(os.path.join(output_dir, f'fold_{fold_id}_loss_curves.png'))
        plt.close()

        # accuracy curves
        plt.figure(figsize=(8, 6))
        plt.plot(history['train_accuracy'], label='Train Acc')
        plt.plot(history['val_accuracy'], label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.title(f'Accuracy curves (Fold {fold_id})')
        plt.savefig(os.path.join(output_dir, f'fold_{fold_id}_accuracy_curves.png'))
        plt.close()

        # other validation metric trends
        epochs_range = range(1, len(history['val_loss']) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, history['val_precision'], label='Val Precision')
        plt.plot(epochs_range, history['val_recall'], label='Val Recall')
        plt.plot(epochs_range, history['val_f1'], label='Val F1')
        plt.plot(epochs_range, history['val_iou'], label='Val IoU')
        plt.plot(epochs_range, history['val_dice'], label='Val Dice')
        plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend()
        plt.title(f'Validation metrics over epochs (Fold {fold_id})')
        plt.savefig(os.path.join(output_dir, f'fold_{fold_id}_val_metrics_trends.png'))
        plt.close()

        # AUC trends
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, [x if x is not None else np.nan for x in history['val_pr_auc']], label='Val PR-AUC')
        plt.plot(epochs_range, [x if x is not None else np.nan for x in history['val_roc_auc']], label='Val ROC-AUC')
        plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.legend()
        plt.title(f'AUC trends (Fold {fold_id})')
        plt.savefig(os.path.join(output_dir, f'fold_{fold_id}_val_auc_trends.png'))
        plt.close()

        # ROC and PR for last epoch of this fold
        try:
            if 'all_val_probs' in locals() and all_val_probs.size > 0 and all_val_targets.size > 0:
                if np.unique(all_val_targets).size > 1:
                    fpr, tpr, _ = roc_curve(all_val_targets, all_val_probs)
                    roc_auc_val = roc_auc_score(all_val_targets, all_val_probs)
                    plt.figure(figsize=(6, 6))
                    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc_val:.4f}')
                    plt.plot([0, 1], [0, 1], linestyle='--')
                    plt.xlabel('FPR'); plt.ylabel('TPR')
                    plt.title(f'ROC Curve (Fold {fold_id}, last val epoch)')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, f'fold_{fold_id}_roc_curve_last_epoch.png'))
                    plt.close()
                precision_vals, recall_vals, _ = precision_recall_curve(all_val_targets, all_val_probs)
                pr_auc_val = auc(recall_vals, precision_vals)
                plt.figure(figsize=(6, 6))
                plt.plot(recall_vals, precision_vals, label=f'PR AUC={pr_auc_val:.4f}')
                plt.xlabel('Recall'); plt.ylabel('Precision')
                plt.title(f'Precision-Recall (Fold {fold_id}, last val epoch)')
                plt.legend()
                plt.savefig(os.path.join(output_dir, f'fold_{fold_id}_pr_curve_last_epoch.png'))
                plt.close()
        except Exception:
            pass

        return model, history

    except Exception as e:
        print(f"Fold {fold_id}: Training error: {e}")
        try:
            torch.save(model.state_dict(), os.path.join(output_dir, f'fold_{fold_id}_emergency_save.pth'))
            print(f"Fold {fold_id}: Saved emergency checkpoint.")
        except Exception:
            pass
        raise

#select data directory for training
def main():
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(title="Select Directory with TEM Images and Masks")
    if not data_dir:
        print("No directory selected. Exiting.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = os.path.join(data_dir, f'nanopore_model_{timestamp}')
    os.makedirs(base_output_dir, exist_ok=True)

    print("\nStarting nanopore detection model training with 3-fold cross-validation")
    print(f"Model checkpoints and outputs will be saved under: {base_output_dir}")

    # group originals (no CV split here)
    original_groups = get_original_and_augmented_groups(data_dir)
    original_images = list(original_groups.keys())
    n_originals = len(original_images)
    print(f"Found {n_originals} original images with augmentations")

    if n_originals < 3:
        print("Not enough original images for 3-fold CV. Need at least 3.")
        return
    
    # Chnage the number of folds here for cross validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42) 

    fold_histories = []
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(original_images), start=1):
        train_originals = [original_images[i] for i in train_idx]
        val_originals = [original_images[i] for i in val_idx]

        fold_output_dir = os.path.join(base_output_dir, f'fold_{fold_id}')
        os.makedirs(fold_output_dir, exist_ok=True)

        print(f"\nStarting Fold {fold_id}/3")
        model, history = train_nanopore_detector( data_dir=data_dir, output_dir=fold_output_dir, train_originals=train_originals, val_originals=val_originals,
                        batch_size=1, epochs=30, learning_rate=5e-5,  patience=3, weight_decay=1e-4, prob_threshold=0.5, resize_to=(1024, 1024),
                     fold_id=fold_id )
        fold_histories.append({ "fold": fold_id, "best_val_loss": min(history["val_loss"]) if len(history["val_loss"]) > 0 else None})

        # free GPU between folds
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # save simple CV summary at base dir
    summary_csv = os.path.join(base_output_dir, "cv_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_val_loss"])
        for fh in fold_histories:
            writer.writerow([fh["fold"], fh["best_val_loss"]])
    print(f"\n3-fold CV finished. Summary saved to: {summary_csv}")


if __name__ == '__main__':
    main()