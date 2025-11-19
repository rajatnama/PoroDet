# Nano-porosities-detection-in-Transmission-electron-microscope-images-using-computer-vision-and-CNN
The Repo contains the training and detection code for nano-porosities present in the Fresnel contrast Transmission electron microscope images using the Unet CNN code architecture.
Overview

This repository contains the full code workflow for contrast-based detection of nanoporous features in Fresnel-contrast transmission electron microscope (TEM) images.
The approach is currently developed for detecting nanoporosities in oxide layers formed on zirconium (Zr) alloys, but the same method can be adapted to other materials and microscopy datasets.

The workflow uses a U-Net convolutional neural network (CNN) to identify nanopores in TEM micrographs, followed by morphological analysis to quantify pore geometry and porosity.
This repository consists of four standalone Python scripts / Jupyter notebooks.

**Each script corresponds to one stage of the nanopore-detection workflow:**

**Data_augmentation.py**
Augments the TEM images and masks through changes in brightness, contrast, rotation, and noise to improve model robustness under different Fresnel-contrast conditions.

**U-net_training.py**
Trains a U-Net CNN using the augmented image–mask pairs.
The network learns to distinguish nanopores from the surrounding oxide microstructure.

**Nanoporosty_detector.py**
Uses the trained model to predict nanopore regions on new TEM images.

**Nanoporosity_analyzer.py**
Performs post-detection analysis.

All four notebooks/scripts can be run independently in Python (≥3.8) or a Jupyter environment.

The code has been written with help of latest models from anthropic and openAI, and we encourage the users to use these AI tools as well to understand to code, set up environment and deal with dependencies.
