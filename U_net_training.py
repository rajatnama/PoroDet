#TEM nanopore detection with Anti-overfitting measures
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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # Add dropout for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.2):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512, dropout_rate)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024, dropout_rate)
        
        # Decoder
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
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        
        # Decoder
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

def get_original_and_augmented_groups(data_dir):
    """Group files by original images and their augmentations"""
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])
    
    # Group by original image
    original_groups = {}
    for file in all_files:
        if "_aug_" in file:
            original_name = file.split("_aug_")[0]
            aug_num = int(file.split("_aug_")[1].split('.')[0])
            
            if original_name not in original_groups:
                original_groups[original_name] = []
            original_groups[original_name].append((file, aug_num))
    
    # Sort augmentations within each group
    for orig in original_groups:
        original_groups[orig].sort(key=lambda x: x[1])
    
    return original_groups

class NanoporeDataset(Dataset):
    def __init__(self, data_dir, original_images, is_validation=False):
        self.data_dir = data_dir
        self.is_validation = is_validation
        
        # Get all TIF files and their corresponding masks
        all_files = sorted(os.listdir(data_dir))
        self.image_files = []
        
        for file in all_files:
            if file.endswith('.tif'):
                # Check if file is from our selected original images
                original_name = file.split("_aug_")[0] if "_aug_" in file else file.replace('.tif', '')
                
                if original_name in original_images:
                    mask_file = file.replace('.tif', '_mask.png')
                    
                    # Only include if mask exists
                    if mask_file in all_files:
                        self.image_files.append(file)
        
        print(f"{'Validation' if is_validation else 'Training'} set contains {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            # Get image and mask paths
            img_path = os.path.join(self.data_dir, self.image_files[idx])
            mask_path = os.path.join(self.data_dir, self.image_files[idx].replace('.tif', '_mask.png'))
            
            # Read images
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                raise ValueError(f"Failed to read image or mask: {self.image_files[idx]}")
            
            # Resize to 1024x1024
            image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0
            
            # Convert to tensors
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
            mask = torch.from_numpy(mask).unsqueeze(0)
            
            return image, mask
        
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a placeholder to avoid crashing
            # In practice, could use a random image from the dataset
            placeholder = torch.zeros((1, 1024, 1024), dtype=torch.float32)
            return placeholder, placeholder

def check_overfitting(train_loss, val_loss, threshold=0.3):
    """Check if model is overfitting based on train/val loss gap"""
    if val_loss > 0 and train_loss < val_loss * (1 - threshold):
        return True
    return False

def train_nanopore_detector(data_dir, output_dir, 
                          batch_size=1, 
                          epochs=30,
                          learning_rate=5e-5,
                          patience=3,
                          weight_decay=1e-4):
    """Train nanopore detector with anti-overfitting measures"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get image groups
    original_groups = get_original_and_augmented_groups(data_dir)
    original_images = list(original_groups.keys())
    print(f"Found {len(original_images)} original images with augmentations")
    
    # Split into train/val by original images
    np.random.seed(42)  # For reproducibility
    val_size = max(2, int(len(original_images) * 0.15))  # 15% for validation
    val_originals = np.random.choice(original_images, size=val_size, replace=False)
    train_originals = [img for img in original_images if img not in val_originals]
    
    print(f"Using {len(train_originals)} originals for training, {len(val_originals)} for validation")
    print(f"Validation originals: {val_originals}")
    
    # Create datasets
    train_dataset = NanoporeDataset(data_dir, train_originals, is_validation=False)
    val_dataset = NanoporeDataset(data_dir, val_originals, is_validation=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model with dropout
    model = UNet(in_channels=1, out_channels=1, dropout_rate=0.2).to(device)
    
    # Loss and optimizer with weight decay
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Setup for training
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("\nStarting training with anti-overfitting measures...")
    print(f"Early stopping patience: {patience}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    
    try:
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
                for batch_idx, (images, masks) in enumerate(pbar):
                    images, masks = images.to(device), masks.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    # Periodically clear cache
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    # Save some validation predictions for visual inspection
                    if epoch % 5 == 0 and val_loss == 0:  # First batch of every 5th epoch
                        output_sigmoid = torch.sigmoid(outputs)
                        for i in range(min(2, len(images))):
                            # Create visualization
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            
                            # Original image
                            axes[0].imshow(images[i].cpu().numpy()[0], cmap='gray')
                            axes[0].set_title('TEM Image')
                            axes[0].axis('off')
                            
                            # Ground truth mask
                            axes[1].imshow(masks[i].cpu().numpy()[0], cmap='gray')
                            axes[1].set_title('True Mask')
                            axes[1].axis('off')
                            
                            # Predicted mask
                            axes[2].imshow(output_sigmoid[i].cpu().numpy()[0], cmap='gray')
                            axes[2].set_title('Predicted Mask')
                            axes[2].axis('off')
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}_sample_{i}.png'))
                            plt.close()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Check for overfitting
            is_overfitting = check_overfitting(avg_train_loss, avg_val_loss)
            
            # Print epoch summary
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}', end='')
            if is_overfitting:
                print(' - WARNING: Potential overfitting detected!')
            else:
                print('')
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save checkpoint every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }, os.path.join(output_dir, 'best_model.pth'))
                print(f'Saved new best model with validation loss: {best_val_loss:.4f}')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f'Validation loss did not improve. Counter: {early_stop_counter}/{patience}')
                
                if early_stop_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            # Clear GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Plot train/val loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
        plt.close()
        
        return model
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        print("Saving emergency checkpoint...")
        torch.save(model.state_dict(), os.path.join(output_dir, 'emergency_save.pth'))
        raise e

def main():
    # Setup UI for folder selection
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(title="Select Directory with TEM Images and Masks")
    
    if not data_dir:
        print("No directory selected. Exiting.")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(data_dir, f'nanopore_model_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStarting nanopore detection model training...")
    print(f"Model checkpoints will be saved to: {output_dir}")
    
    try:
        model = train_nanopore_detector(data_dir, output_dir)
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"\nTraining complete. Final model saved to: {final_model_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        if torch.cuda.is_available():
            print(f"GPU Memory at failure: {torch.cuda.memory_allocated()/1e9:.2f} GB")

if __name__ == '__main__':
    main()