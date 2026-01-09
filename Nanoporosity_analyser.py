import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Load UNet mask and resize to 1024x1024, keep the image patch size consistent
def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask ({mask_path})")
    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    return mask

# Analyze nanopores using ellipse fitting for aspect ratio 
def analyze_nanopores(mask, threshold=200, aspect_ratio_threshold=1.5):
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    labeled_mask, num_features = label(binary_mask)
    total_pixels = binary_mask.size
    nanopore_pixels = np.sum(binary_mask > 0)
    porosity_percentage = (nanopore_pixels / total_pixels) * 100
    
    sizes = []
    aspect_ratios = []
    diameters = []
    circular_count = 0
    elongated_count = 0
    circular_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    elongated_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    
    for i in range(1, num_features + 1):
        nanopore = (labeled_mask == i).astype(np.uint8)
        area = np.sum(nanopore)
        sizes.append(area)
        
        contours, _ = cv2.findContours(nanopore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = contours[0]
        
        # Fit an ellipse to the contour (requires at least 5 points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 1.0
        else:
            # Fallback to bounding rectangle if contour is too small
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) != 0 else 1.0
        
        aspect_ratios.append(aspect_ratio)
        
        if aspect_ratio <= aspect_ratio_threshold:
            circular_count += 1
            circular_mask |= nanopore
        else:
            elongated_count += 1
            elongated_mask |= nanopore
        
        diameter = 2 * np.sqrt(area / np.pi)
        diameters.append(diameter)
    
    return porosity_percentage, sizes, aspect_ratios, diameters, circular_count, elongated_count, circular_mask, elongated_mask

# Create overlay images for circular and elongated nanopores
def create_overlays(mask, circular_mask, elongated_mask, output_dir, image_name):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    blue = [255, 0, 0]  # Blue for circular
    red = [0, 0, 255]   # Red for elongated
    
    circular_overlay = mask_rgb.copy()
    circular_overlay[circular_mask > 0] = blue
    circular_path = os.path.join(output_dir, f'{image_name}_circular_overlay.png')
    cv2.imwrite(circular_path, circular_overlay)
    print(f"Circular overlay saved to: {circular_path}")
    
    elongated_overlay = mask_rgb.copy()
    elongated_overlay[elongated_mask > 0] = red
    elongated_path = os.path.join(output_dir, f'{image_name}_elongated_overlay.png')
    cv2.imwrite(elongated_path, elongated_overlay)
    print(f"Elongated overlay saved to: {elongated_path}")

# Generate and save histograms with 0.5 increments on x-axis
def plot_histograms(sizes, aspect_ratios, diameters, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    size_min, size_max = min(sizes), max(sizes)
    size_bins = np.arange(np.floor(size_min / 0.5) * 0.5, np.ceil(size_max / 0.5) * 0.5 + 0.5, 0.5)
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=size_bins, color='blue', edgecolor='black')
    plt.title('Histogram of Nanopore Sizes (Area in Pixels)')
    plt.xlabel('Size in pixels')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    size_path = os.path.join(output_dir, f'{image_name}_size_histogram.png')
    plt.savefig(size_path)
    plt.close()
    print(f"Size histogram saved to: {size_path}")
    
    ar_min, ar_max = min(aspect_ratios), max(aspect_ratios)
    ar_bins = np.arange(np.floor(ar_min / 0.5) * 0.5, np.ceil(ar_max / 0.5) * 0.5 + 0.5, 0.5)
    plt.figure(figsize=(10, 6))
    plt.hist(aspect_ratios, bins=ar_bins, color='orange', edgecolor='black')
    plt.title('Histogram of Nanopore Aspect Ratios')
    plt.xlabel('Aspect Ratio (≥ 1)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    aspect_path = os.path.join(output_dir, f'{image_name}_aspect_ratio_histogram.png')
    plt.savefig(aspect_path)
    plt.close()
    print(f"Aspect ratio histogram saved to: {aspect_path}")
    
    dia_min, dia_max = min(diameters), max(diameters)
    dia_bins = np.arange(np.floor(dia_min / 0.5) * 0.5, np.ceil(dia_max / 0.5) * 0.5 + 0.5, 0.5)
    plt.figure(figsize=(10, 6))
    plt.hist(diameters, bins=dia_bins, color='green', edgecolor='black')
    plt.title('Histogram of Nanopore Equivalent Diameters in Pixels')
    plt.xlabel('Diameter (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    diameter_path = os.path.join(output_dir, f'{image_name}_diameter_histogram.png')
    plt.savefig(diameter_path)
    plt.close()
    print(f"Diameter histogram saved to: {diameter_path}")

# Process one mask to analyze nanoporosity, shapes, and generate overlays
def process_single_mask(mask_path, output_dir, aspect_ratio_threshold=1.5):

    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(mask_path))[0].replace('_mask', '')
    
    try:
        mask = load_mask(mask_path)
        porosity, sizes, aspect_ratios, diameters, circular_count, elongated_count, circular_mask, elongated_mask = analyze_nanopores(
            mask, aspect_ratio_threshold=aspect_ratio_threshold
        )
        
        print(f"{image_name}.tif: Nanoporosity = {porosity:.2f}%")
        print(f"Total number of nanopores detected: {len(sizes)}")
        print(f"Number of mostly circular nanopores (aspect ratio ≤ {aspect_ratio_threshold}): {circular_count}")
        print(f"Number of elongated nanopores (aspect ratio > {aspect_ratio_threshold}): {elongated_count}")
        print(f"Average size: {np.mean(sizes):.2f} pixels")
        print(f"Average aspect ratio: {np.mean(aspect_ratios):.2f}")
        print(f"Average diameter: {np.mean(diameters):.2f} pixels")
        
        plot_histograms(sizes, aspect_ratios, diameters, output_dir, image_name)
        create_overlays(mask, circular_mask, elongated_mask, output_dir, image_name)
        
    except Exception as e:
        print(f"Error processing {image_name}_mask.png: {e}")

def main():
    root = tk.Tk()
    root.withdraw()
    
    mask_path = filedialog.askopenfilename(
        title="Select UNet Mask File",
        filetypes=[("PNG files", "*.png")]
    )
    if not mask_path:
        print("No mask selected. Exiting.")
        return
    
    output_dir = filedialog.askdirectory(title="Select Output Directory for Analysis")
    if not output_dir:
        print("No output directory selected. Exiting.")
        return
    
    aspect_ratio_threshold = 2  # Adjustable
    
    print(f"Processing mask: {mask_path}")
    print(f"Saving analysis to: {output_dir}")
    process_single_mask(mask_path, output_dir, aspect_ratio_threshold=aspect_ratio_threshold)
    print("Processing complete.")

if __name__ == '__main__':
    main()