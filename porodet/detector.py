# TEM nanoporosity detection
# Import necessary libraries 
import os
import numpy as np
import torch
import cv2
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# Import the UNet model architecture
from .model import UNet

# Load the trained the model
def load_model(model_path):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Load weights
    # Added weights_only=False for PyTorch 2.6+ compatibility
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} "
              f"with validation loss {checkpoint.get('val_loss', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state (no metadata available)")
    
    model.eval()
    return model, device

# Predict nanoporosity in the new TEM image
def predict_nanopores(model, image_path, device, threshold=0.5):   # adjust the threshold to detect the more or less pores as per probability map 
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = image.copy()
    
    # Resize to 1024x1024 for prediction
    # Keep the image size as in the training
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        prediction = prediction.cpu().numpy()[0, 0]
    
    # Apply threshold
    binary_prediction = (prediction > threshold).astype(np.float32)
    
    # Resize back to original dimensions
    h, w = original_image.shape
    prediction_resized = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)
    binary_prediction_resized = cv2.resize(binary_prediction, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return original_image, prediction_resized, binary_prediction_resized

# Analyze nanopore statistics from binary mask
def analyze_nanopores(binary_mask):
    contours, _ = cv2.findContours(
        (binary_mask * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Calculate nanopore metrics
    num_pores = len(contours)
    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    
    # Filter out very small contours
    min_area = 5  # Adjust as needed
    valid_areas = [a for a in areas if a > min_area]
    
    if len(valid_areas) > 0:
        avg_area = np.mean(valid_areas)
        total_area = np.sum(valid_areas)
        porosity = total_area / (binary_mask.shape[0] * binary_mask.shape[1]) * 100
    else:
        avg_area = 0
        total_area = 0
        porosity = 0
    
    return {
        'num_pores': num_pores,
        'avg_area': avg_area,
        'total_area': total_area,
        'porosity': porosity
    }

# Create overlay of nanopore prediction on original image
def create_overlay(image, prediction, threshold=0.5, alpha=0.6):
    # Create RGB version of grayscale image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply color to prediction
    binary_mask = (prediction > threshold)
    
    # Create heatmap of prediction
    heatmap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    heatmap[binary_mask] = [255, 0, 0]  # Red for detected nanopores
    
    # Blend original image with prediction overlay
    overlay = cv2.addWeighted(rgb_image, 1, heatmap, alpha, 0)
    
    return overlay

# Create and save visualization of nanopore detection result
def visualize_results(original_image, prediction, binary_prediction, metrics, output_path):
    # Normalize images for visualization
    original_norm = original_image.astype(np.float32) / 255
    
    # Create overlay image
    overlay = create_overlay(original_image, prediction)
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_norm, cmap='gray')
    plt.title('Original TEM Image')
    plt.axis('off')
    
    # Prediction heatmap
    plt.subplot(2, 2, 2)
    plt.imshow(prediction, cmap='hot')
    plt.colorbar(label='Nanopore Probability')
    plt.title('Nanopore Prediction probability Heatmap')
    plt.axis('off')
    
    # Binary prediction
    plt.subplot(2, 2, 3)
    plt.imshow(binary_prediction, cmap='binary')
    plt.title('Binary Nanopore Mask')
    plt.axis('off')
    
    # Overlay
    plt.subplot(2, 2, 4)
    plt.imshow(overlay)
    plt.title('Nanoporosity Overlay on TEM Image')
    plt.axis('off')
    
    # Add metrics
    plt.figtext(0.5, 0.02, 
                f"Total nanopores: {metrics['num_pores']}\n"
                f"Average area: {metrics['avg_area']:.2f} pixels\n"
                f"Porosity: {metrics['porosity']:.2f}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Results saved to: {output_path}")

# Save binary prediction as a mask image
def save_mask_image(binary_prediction, output_path):
    cv2.imwrite(output_path, (binary_prediction * 255).astype(np.uint8))
    print(f"Binary mask saved to: {output_path}")

def main():
    # Select model file
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(
        title="Select Trained Model File", 
        filetypes=[("PyTorch Model", "*.pth")]
    )
    
    if not model_path:
        print("No model selected. Exiting.")
        return
    
    # Select TEM image
    image_path = filedialog.askopenfilename(
        title="Select TEM Image for Analysis",
        filetypes=[("Image files", "*.tif *.png *.jpg")]
    )
    
    if not image_path:
        print("No image selected. Exiting.")
        return
    
    # Create output directory
    image_dir = os.path.dirname(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(image_dir, f'nanopore_analysis_{image_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initialozing the nanopore analysis for: {image_name}")
    print(f"Results will be saved to: {output_dir}")
    
    # Load model
    model, device = load_model(model_path)
    
    # Predict nanopores
    original, prediction, binary_prediction = predict_nanopores(model, image_path, device)
    
    # Analyze nanopore statistics
    metrics = analyze_nanopores(binary_prediction)
    print("\nNanopore Analysis Results:")
    print(f"  Total nanopores detected: {metrics['num_pores']}")
    print(f"  Average nanopore area: {metrics['avg_area']:.2f} pixels")
    print(f"  Total nanopore area: {metrics['total_area']:.2f} pixels")
    print(f"  Porosity: {metrics['porosity']:.2f}%")
    
    # Create and save visualization
    viz_path = os.path.join(output_dir, f"{image_name}_analysis.png")
    visualize_results(original, prediction, binary_prediction, metrics, viz_path)
    
    # Save binary mask
    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
    save_mask_image(binary_prediction, mask_path)
    
    # Save metrics to text file
    metrics_path = os.path.join(output_dir, f"{image_name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Nanopore Analysis Results\n")
        f.write(f"Image: {image_name}\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total nanopores detected: {metrics['num_pores']}\n")
        f.write(f"Average nanopore area: {metrics['avg_area']:.2f} pixels\n")
        f.write(f"Total nanopore area: {metrics['total_area']:.2f} pixels\n")
        f.write(f"Porosity: {metrics['porosity']:.2f}%\n")
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")

if __name__ == '__main__':
    main()