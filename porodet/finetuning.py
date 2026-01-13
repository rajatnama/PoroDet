"""
PoroDet_finetune.py

Fine-tune a pretrained PoroDet U-Net model on new TEM image–mask pairs.

Expected data layout in the chosen folder:

    Image_01.tif
    Image_01_mask.png
    Image_02.tif
    Image_02_mask.png
    ...

Masks must be PNGs with the same base filename + "_mask".
"""

import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

# Import the unet model architecture
from .model import UNet

# Training parameters

image_size = 1024          # change to 2048 if you really want (watch GPU memory)
batch_size = 1
epochs = 10
learning_rate = 5e-5
weight_decay = 1e-4
patience = 3               # early stopping if val loss doesn't improve
freeze_encoder = False     # True = only fine-tune decoder


# Dataset

class NanoporeDataset(Dataset):
    """
    Simple dataset for image + mask pairs in a single folder.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        all_files = sorted(os.listdir(data_dir))

        self.image_files = []
        for fname in all_files:
            if fname.lower().endswith(".tif"):
                base = os.path.splitext(fname)[0]
                mask_name = base + "_mask.png"
                if mask_name in all_files:
                    self.image_files.append(fname)

        print(f"Found {len(self.image_files)} image/mask pairs in {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base = os.path.splitext(img_name)[0]
        mask_name = base + "_mask.png"

        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Could not read image or mask: {img_name}")

        # resize
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        # normalise
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # to (C, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# Helper functions

def choose_file(title, pattern="*.pth"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=[("Files", pattern), ("All files", "*.*")])
    root.destroy()
    return path


def choose_dir(title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


def load_pretrained_model(path, device):
    print(f"\nLoading pretrained model from:\n  {path}")
    model = UNet(in_channels=1, out_channels=1, dropout_rate=0.2).to(device)

    checkpoint = torch.load(path, map_location=device)
    # support both "training checkpoint" dict and raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model_state_dict (epoch={checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict directly (no metadata).")

    return model


def freeze_encoder_part(model: UNet):
    """
    Freeze encoder and bottleneck so only the decoder is updated.
    """
    print("Freezing encoder + bottleneck layers.")
    encoder_modules = [
        model.conv1, model.pool1,
        model.conv2, model.pool2,
        model.conv3, model.pool3,
        model.conv4, model.pool4,
        model.conv5,
    ]
    for m in encoder_modules:
        for p in m.parameters():
            p.requires_grad = False



# Metrics

def compute_pixel_metrics(y_true, y_prob, threshold=0.5):
    """
    y_true, y_prob: 1D numpy arrays over all pixels in the validation set.
    Returns dict with accuracy, precision, recall, F1, IoU, Dice, PR-AUC, ROC-AUC.
    """
    y_pred = (y_prob >= threshold).astype(np.uint8)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # AUC metrics are undefined if only one class present
    if np.unique(y_true).size > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = float("nan")
        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc = float("nan")
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


# Train / eval loops


def train_epoch(model, loader, criterion, optimizer, device, threshold=0.5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # training accuracy (per pixel)
        probs = torch.sigmoid(out).detach().cpu().numpy().ravel()
        labels = masks.detach().cpu().numpy().ravel()
        preds = (probs >= threshold).astype(np.uint8)

        correct += np.sum(preds == labels)
        total += labels.size

    avg_loss = running_loss / max(1, len(loader))
    train_acc = correct / total if total > 0 else 0.0
    return avg_loss, train_acc


def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_true = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)
            loss = criterion(out, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(out).cpu().numpy().ravel()
            labels = masks.cpu().numpy().ravel()

            all_probs.append(probs)
            all_true.append(labels)

    avg_loss = running_loss / max(1, len(loader))
    if all_probs:
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_true)
        metrics = compute_pixel_metrics(y_true, y_prob, threshold=threshold)
    else:
        y_prob = np.array([])
        y_true = np.array([])
        metrics = compute_pixel_metrics(y_true, y_prob, threshold=threshold)

    return avg_loss, metrics, y_true, y_prob


# Main fine-tune routine

def main():
    # pick pretrained model
    pretrained_path = choose_file("Select pretrained PoroDet model (.pth)", "*.pth")
    if not pretrained_path:
        print("No model selected. Exiting.")
        return

    # pick new data
    data_dir = choose_dir("Select folder with NEW TEM images and masks")
    if not data_dir:
        print("No data folder selected. Exiting.")
        return

    # pick output folder
    out_root = choose_dir("Select output folder for fine-tuning results")
    if not out_root:
        print("No output folder selected. Exiting.")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"porodet_finetune_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutputs will be saved in:\n  {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # dataset
    dataset = NanoporeDataset(data_dir)
    if len(dataset) == 0:
        print("No valid image/mask pairs found. Exiting.")
        return

    # simple 80/20 split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train images: {len(train_set)}, Val images: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # model + optimiser
    model = load_pretrained_model(pretrained_path, device)

    if freeze_encoder:
        freeze_encoder_part(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_val = float("inf")
    no_improve = 0

    # for logging
    epochs_list = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_ious = []
    val_dices = []
    val_pr_aucs = []
    val_roc_aucs = []

    last_y_true = None
    last_y_prob = None

    print("\nStarting fine-tuning...")
    print(f"  epochs        : {epochs}")
    print(f"  learning rate : {learning_rate}")
    print(f"  weight decay  : {weight_decay}")
    print(f"  freeze encoder: {freeze_encoder}")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics, y_true, y_prob = eval_epoch(
            model, val_loader, criterion, device
        )

        # store last epoch preds for PR/ROC plots
        last_y_true = y_true
        last_y_prob = y_prob

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_metrics["accuracy"])
        val_precs.append(val_metrics["precision"])
        val_recs.append(val_metrics["recall"])
        val_f1s.append(val_metrics["f1"])
        val_ious.append(val_metrics["iou"])
        val_dices.append(val_metrics["dice"])
        val_pr_aucs.append(val_metrics["pr_auc"])
        val_roc_aucs.append(val_metrics["roc_auc"])

        print(f"  train loss: {train_loss:.4f} | train acc: {train_acc:.4f}")
        print(
            "  val   loss: {:.4f} | val acc: {:.4f} | prec: {:.3f} | "
            "rec: {:.3f} | F1: {:.3f} | IoU: {:.3f} | Dice: {:.3f} | "
            "PR-AUC: {:.3f} | ROC-AUC: {:.3f}".format(
                val_loss,
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["iou"],
                val_metrics["dice"],
                val_metrics["pr_auc"],
                val_metrics["roc_auc"],
            )
        )

        scheduler.step(val_loss)

        # always save "last epoch"
        last_path = os.path.join(out_dir, "finetuned_last_epoch.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "finetuned_from": os.path.basename(pretrained_path),
            },
            last_path,
        )

        # check for new best
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            no_improve = 0
            best_path = os.path.join(out_dir, "finetuned_best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "finetuned_from": os.path.basename(pretrained_path),
                },
                best_path,
            )
            print(f"  → New best model saved: {best_path}")
        else:
            no_improve += 1
            print(f"  → No improvement for {no_improve} epoch(s).")

        if no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # Save metrics to CSV
    
    import csv

    metrics_csv = os.path.join(out_dir, "finetune_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_iou",
            "val_dice",
            "val_pr_auc",
            "val_roc_auc",
        ])
        for i in range(len(epochs_list)):
            writer.writerow([
                epochs_list[i],
                train_losses[i],
                train_accs[i],
                val_losses[i],
                val_accs[i],
                val_precs[i],
                val_recs[i],
                val_f1s[i],
                val_ious[i],
                val_dices[i],
                val_pr_aucs[i],
                val_roc_aucs[i],
            ])
    print(f"\nMetrics saved to:\n  {metrics_csv}")

    
    # Plots: loss + accuracy

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, train_losses, label="train")
    plt.plot(epochs_list, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(loss_plot, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, train_accs, label="train")
    plt.plot(epochs_list, val_accs, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_plot = os.path.join(out_dir, "accuracy_curves.png")
    plt.savefig(acc_plot, dpi=200)
    plt.close()

    print(f"Loss curves saved to:\n  {loss_plot}")
    print(f"Accuracy curves saved to:\n  {acc_plot}")


    # PR & ROC curves for last epoch
    if last_y_true is not None and last_y_true.size > 0 and np.unique(last_y_true).size > 1:
        # PR curve
        precision, recall, _ = precision_recall_curve(last_y_true, last_y_prob)
        plt.figure(figsize=(5, 4))
        plt.step(recall, precision, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Validation PR curve (last epoch)")
        plt.tight_layout()
        pr_plot = os.path.join(out_dir, "pr_curve_last_epoch.png")
        plt.savefig(pr_plot, dpi=200)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(last_y_true, last_y_prob)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Validation ROC curve (last epoch)")
        plt.tight_layout()
        roc_plot = os.path.join(out_dir, "roc_curve_last_epoch.png")
        plt.savefig(roc_plot, dpi=200)
        plt.close()

        print(f"PR curve saved to:\n  {pr_plot}")
        print(f"ROC curve saved to:\n  {roc_plot}")
    else:
        print("\nSkipping PR/ROC plots (validation set had only one class or no data).")

    print("\nFine-tuning finished.")


if __name__ == "__main__":
    main()
