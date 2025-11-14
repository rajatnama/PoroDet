# Nano-porosities-detection-in-Transmission-electron-microscopic-images-using-computer-vision-and-CNN
The Repo contains the training and detection code for nano-porosities present in the Fresnel contrast Transmission electron microscope images using the Unet CNN code architecture.
Overview

This repository contains the full code workflow for contrast-based detection of nanoporous features in Fresnel-contrast transmission electron microscope (TEM) images.
The approach is currently developed for detecting nanoporosities in oxide layers formed on zirconium (Zr) alloys, but the same method can be adapted to other materials and microscopy datasets.

The workflow uses a U-Net convolutional neural network (CNN) to identify nanopores in TEM micrographs, followed by morphological analysis to quantify pore geometry and porosity.
This repository consists of four standalone Python scripts / Jupyter notebooks.

**Each script corresponds to one stage of the nanopore-detection workflow:**

**tem_image_augmentation.py**
Augments the TEM images and masks through changes in brightness, contrast, rotation, and noise to improve model robustness under different Fresnel-contrast conditions.

**train_nanopore_detector.py**
Trains a U-Net CNN using the augmented image–mask pairs.
The network learns to distinguish nanopores from the surrounding oxide microstructure.

**nanopore_detector.py**
Uses the trained model to predict nanopore regions on new TEM images.

**nanopore_analyzer.py**
Performs post-detection analysis.

All four notebooks/scripts can be run independently in Python (≥3.8) or a Jupyter environment.

**How It Works**

**Inputs**: TEM images (Fresnel-contrast mode) and user-defined binary masks marking nanopores.

**Augmentation**: Each image–mask pair is augmented (15 transformations × 20 original images → ~640 total pairs).

**Training**: A U-Net model is trained using the augmented dataset.

**Detection**: New TEM images are analyzed to generate probability maps and binary pore masks.

**Analysis**: Detected pores are characterized by aspect ratio, size, and porosity.

This workflow allows contrast-based nanopore detection and basic quantitative analysis of microstructural features such as pore shape and interconnectivity.
