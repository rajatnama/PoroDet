# Data Augmentation pipeline with albumentation for data diversifation 
# Imports libraries and dependencies
# tkinter is used for data seletction and storage dialog boxes locally, for google colab this can be modified to use google drive mount points
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import albumentations as A
from datetime import datetime
import time

# Define the augmention pipeline and apply different augmentation transformations 
def augmentation_transformations():
    augmentations = [
        # Geometric transformations
        A.Compose([ A.HorizontalFlip(p=1.0),A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        A.Compose([A.Rotate(limit=45, p=1.0),A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]), #adjust and optimize the rotation according to the data 
        A.Compose([A.Rotate(limit=90, p=1.0), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        A.Compose([A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=1.0), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        
        # Intensity transformations
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)]), #adjust and optimize the brightness according to the data 
        A.Compose([A.RandomGamma(gamma_limit=(80, 120), p=1.0)]),
        A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)]),
        
        # Noise and blur inclusion
        A.Compose([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)]),
        A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)]),
        
        # shape Combinations transformation 
        A.Compose([A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        A.Compose([A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=1.0),A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        A.Compose([A.RandomCrop(height=3840, width=3840, p=1.0), A.Resize(height=4096, width=4096, p=1.0),      #crop the image as per need Resize back to original
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0), A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0), A.GaussNoise(var_limit=(5.0, 30.0), p=0.5)])]
    
    return augmentations

def augment_images(input_dir, output_dir, num_augmentations=15): # Optimize the number of augmentation for better training and generalization

    # call augmentation transformations 
    augmentations = augmentation_transformations()
    
    # load all raw images files and its corresponding masks
    all_files = sorted(os.listdir(input_dir))
    image_files = [f for f in all_files if f.endswith('.tif')]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all images
    print(f"Found {len(image_files)} raw images")
    print(f"Generating {num_augmentations} augmentations for each image")
    
    for image_file in tqdm(image_files, desc="Processing Images"):
        image_name = os.path.splitext(image_file)[0]
        
        # Check if binary mask images exists
        mask_file = f"{image_name}_mask.png"
        if mask_file not in all_files:
            print(f"Warning: Mask image not found for {image_file}, skipping")
            continue
        
        # Read raw and mask images
        image_path = os.path.join(input_dir, image_file)
        mask_path = os.path.join(input_dir, mask_file)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"Error reading raw and mask images: {image_file}")
            continue
        
        # Save original image and mask to output directory
        cv2.imwrite(os.path.join(output_dir, image_file), image)
        cv2.imwrite(os.path.join(output_dir, mask_file), mask)
        
        # Create augmentations
        for aug_idx in range(num_augmentations):
            # Select augmentation
            transform = augmentations[aug_idx % len(augmentations)]
            
            # Apply augmentation
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Neme the augmented images
            aug_image_filename = f"{image_name}_aug_{aug_idx+1}.tif"
            aug_mask_filename = f"{image_name}_aug_{aug_idx+1}_mask.png"
            
            # Save augmented raw and mask images
            cv2.imwrite(os.path.join(output_dir, aug_image_filename), aug_image)
            cv2.imwrite(os.path.join(output_dir, aug_mask_filename), aug_mask)
    
    print(f"Augmentation complete. Files saved to: {output_dir}")

# try multiple methods for input directory if chnages syatem to system
def get_input_directory():
    input_dir = None
    # Method 1: Try using tkinter dialog
    try:
        root = tk.Tk()
        root.withdraw()
        
        print("Waiting for folder selection dialog to appear")
        print("If no dialog appears, check your taskbar or behind other windows.")
        print("Dialog will timeout in 15 seconds if not used.")
        
        # Force window to front
        root.attributes('-topmost', True)
        root.update()
        
        # Start timer for dialog
        start_time = time.time()
        input_dir = None
        
        # Give the dialog 15 seconds to be used
        while time.time() - start_time < 15 and input_dir is None: # dialog timing can be adjusted as per need, we used 15 sec
            input_dir = filedialog.askdirectory(title="Select Directory with Original raw TEM and Masks Images")
            root.update_idletasks()
            root.update()
            time.sleep(0.1)
            
            # Break the loop after after directory selection
            if input_dir:
                break
    except Exception as e:
        print(f"GUI dialog failed: {e}")
    
    # Method 2: If GUI failed or timed out, use console input
    if not input_dir:
        print("\nFolder dialog failed or timed out.")
        print("Please enter the full path to your images directory:")
        input_dir = input("Path: ").strip()
    
    # Verify directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist.")
        return None
    
    return input_dir

def main():
    print("TEM Image Augmentation")
    
    # Get input directory
    input_dir = get_input_directory()
    
    if not input_dir:
        print("No valid directory provided.")
        return
    
    print(f"You Selected directory: {input_dir}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.path.dirname(input_dir), f'augmented_images_{timestamp}')
    
    # Set number of augmentations per image
    num_augmentations = 10
    print(f"{num_augmentations} augmentations per image will be created")
    
    # Perform augmentation
    augment_images(input_dir, output_dir, num_augmentations)

if __name__ == '__main__':
    main()