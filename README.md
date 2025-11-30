# Nano-porosity detection in TEM images using U-Net and computer vision

This repository contains the full workflow for contrast-based detection and analysis of nanoporosities in Fresnel-contrast transmission electron microscope (TEM) images using a U-Net convolutional neural network (CNN).

The current implementation was developed for oxides formed on zirconium (Zr) alloys, but the same method can be adapted to other materials and microscopy datasets where pores and cracks appear as bright/dark Fresnel features.

---

## Overview of the workflow

The pipeline has four main stages:

1. **Data augmentation** – increase dataset diversity by modifying contrast, brightness, orientation and noise.
2. **U-Net training** – train a pixel-wise segmentation model that detects nanopores / nanocracks.
3. **Nanoporosity detection** – apply the trained model to new TEM images to obtain pore probability maps and binary masks.
4. **Post-processing and analysis** – classify features as pores vs cracks and quantify nanoporosity and size/shape distributions.

Each stage is implemented as a standalone Python script.

---

## Repository contents

- `Data_augmentation.py`  
  Augments TEM images and corresponding binary masks to improve robustness to variations in Fresnel contrast.  
  - Expects grayscale `.tif` images with matching `*_mask.png` binary masks.  
  - Applies flips, rotations, brightness/contrast adjustments, gamma/CLAHE, noise, blur and mild geometric distortions.  
  - Saves original and augmented image–mask pairs into a new timestamped output directory.

- `U_net_training.py`  
  Trains a U-Net CNN on the (augmented) image–mask pairs for pixel-wise nanopore segmentation.  
  - Resizes images and masks to 1024×1024 pixels.  
  - Uses `BCEWithLogitsLoss` and reports training/validation loss and accuracy, as well as segmentation metrics (precision, recall, F1/Dice, IoU, PR–AUC, ROC–AUC).  
  - Performs K-fold cross-validation over the original images (default K = 3).  
  - Saves best model checkpoints, per-fold metric CSV files and plots (loss curves, accuracy curves, PR/ROC curves).

- `Nanoporosity_detector.py`  
  Applies a trained U-Net model to a new TEM image.  
  - Loads a `.pth` checkpoint produced by `U_net_training.py`.  
  - Produces a nanopore probability map, a binary nanopore mask and an overlay on the original TEM image.  
  - Computes basic statistics such as total pore count, total pore area and nanoporosity (%) for the analysed image.  
  - Saves a composite figure (`<image_name>_analysis.png`), the binary mask (`<image_name>_mask.png`) and a metrics text file.

- `Nanoporosity_analyser.py`  
  Performs detailed post-processing on a UNet-generated pore mask.  
  - Loads the binary mask, labels individual features and measures area, aspect ratio (from ellipse fitting) and equivalent diameter.  
  - Classifies objects as *nanopores* (approximately circular, low aspect ratio) or *nanocracks* (elongated, high aspect ratio) based on a user-defined aspect-ratio threshold.  
  - Computes total nanoporosity percentage and the number of pores vs cracks.  
  - Saves histograms of size, aspect ratio and diameter, as well as coloured overlays highlighting pores and cracks separately.

- `PoroDet_finetune.py`  
  Fine-tunes an **existing** PoroDet U-Net model (`.pth`) on a new set of TEM images and masks.  
  - Starts from a pretrained checkpoint (e.g. the Zr-oxide model)  
  - Optionally freezes the encoder and only updates the decoder  
  - Reports the same metrics as the main trainer (loss, accuracy, precision, recall, F1, IoU, Dice, PR-AUC, ROC-AUC)  
  - Saves `finetuned_best_model.pth`, `finetuned_last_epoch.pth`, and a `finetune_metrics.csv` file plus loss/accuracy and PR/ROC plots

---
We acknowledge the use of large-language models (Anthropic, OpenAI) as coding assistants during development of this software, and we encourage users to use similar tools to help understand the code, set up their Python environments, and troubleshoot dependencies.

## Installation

Create a Python environment (example using conda):

```bash
conda create -n porodet python=3.10
conda activate porodet

pip install -r requirements.txt


