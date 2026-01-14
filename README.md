# PoroDet: Nano-porosity detection in TEM images using U-Net and computer vision

**PoroDet** is a Python package for the contrast-based detection and analysis of nanoporosities in Fresnel-contrast transmission electron microscope (TEM) images. It utilizes a U-Net convolutional neural network (CNN) to segment and quantify pores and cracks.

The current implementation was developed for oxides formed on zirconium (Zr) alloys, but the same method can be adapted to other materials and microscopy datasets where pores and cracks appear as bright/dark Fresnel features.

---

## Installation

### Step 1: Installing the package

### Option 1: Install directly from GitHub (Recommended for Colab)
Install the package directly into your environment using pip:

pip install git+[https://github.com/Deep7285/PoroDet.git](https://github.com/Deep7285/PoroDet.git)

### Option 2 : Clone directly from GitHub and Install Locally
  1. Clone the repository:
  git clone [https://github.com/Deep7285/PoroDet.git](https://github.com/Deep7285/PoroDet.git)
  cd PoroDet

  2. Create environment (optional but recommended): 
  (Visit the conda website to create the enviroment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
  ```bash
  conda create -n porodet python=3.10 
  conda activate porodet 
  ```
  3. Install the package: 
  ```bash
  pip install .
  ```

## Usage
Once installed, the package can be imported (import porodet) in any Python script or Jupyter Notebook. The package contain five main tools:

## Step 2:  Data Augmentation
Increase dataset diversity by applying augmentation on input TEM images and corresponding binary masks to improve robustness to variations in Fresnel contrast.  
  - Expects grayscale `.tif` images with matching `.png` binary masks.  
  - Applies flips, rotations, brightness/contrast adjustments, gamma/CLAHE, noise, blur and mild geometric distortions.  
  - Saves original and augmented image–mask pairs into a new timestamped output directory.
```bash
import porodet
# This triggers the GUI for folder selection
porodet.augment()
```
Note: 
  1. By default the the pacake resize the input image into 1024X1024 pixels. 
  2. Keep the square size input images. For example if input images are 4096X1024 pixel then image will be distored during augmentation and    training performance will be affected.

## Step 3: Training the U-net model
Trains a U-Net CNN on the image–mask pairs for pixel-wise nanopore segmentation.  
  - Resizes images and masks to 1024×1024 pixels.  
  - Uses `BCEWithLogitsLoss` and reports training/validation loss and accuracy, as well as segmentation metrics (precision, recall, F1/Dice, IoU, PR–AUC, ROC–AUC).  
  - Performs K-fold cross-validation over the original images (default K = 3).  
  - Saves best model checkpoints, per-fold metric CSV files and plots (loss curves, accuracy curves, PR/ROC curves).
```bash
# Starts the training workflow
porodet.train()
```
## Step 4: Nanoporosities detetions (inferences)
Applies a trained U-Net model to a new TEM image.  
  - Loads a `.pth` trained model.  
  - Produces a nanopore probability map, a binary nanopore mask and an overlay on the original TEM image.  
  - Computes basic statistics such as total pore count, total pore area and nanoporosity (%) for the analysed image.  
  - Saves a composite figure (`<image_name>_analysis.png`), the binary mask (`<image_name>_mask.png`) and a metrics text file.
```bash
# Select your model and target image via GUI
porodet.detect()
```

## Step 5: Analysis
Performs detailed post-processing on a UNet-generated pore mask.  
  - Loads the binary mask, labels individual features and measures area, aspect ratio (from ellipse fitting) and equivalent diameter.  
  - Classifies objects as *nanopores* (approximately circular, low aspect ratio) or *nanocracks* (elongated, high aspect ratio) based on a user-defined aspect-ratio threshold.  
  - Computes total nanoporosity percentage and the number of pores vs cracks.  
  - Saves histograms of size, aspect ratio and diameter, as well as coloured overlays highlighting pores and cracks separately.
```bash
# process a single mask file
porodet.analyze()
```

## Additional Step if need to fine tune the trained model or fine tune the pre-trained model
Fine-tunes an **existing** PoroDet U-Net model (`.pth`) on a new set of TEM images and masks and and allows freezing the encoder layers to adapt the model to new materials with smaller  datasets.
  - Starts from a pretrained checkpoint (e.g. the Zr-oxide model)  
  - Optionally freezes the encoder and only updates the decoder  
  - Reports the same metrics as the main trainer (loss, accuracy, precision, recall, F1, IoU, Dice, PR-AUC, ROC-AUC)  
```bash
# Select pretrained model and new dataset
porodet.finetune()
```
---
## Development Note
This software was developed with the assistance of large language models (LLMs) for code generation and debugging support.
Any contribution or suggestions to the packege are welcome.

## Authers
Rajat Nama (University of Oxford)
  - github: https://github.com/rajatnama
  - web: https://nanoanalysis.web.ox.ac.uk/people/rajat-nama

Deepak Kumar (Indian Institute of Technology Madras, India)
  - github: https://github.com/Deep7285
  - web: https://sites.google.com/view/deepak7285/home

## License
This project is licensed under the MIT License. See the LICENSE file for details.




