# import initiator for porodet package


# Import the model architecture
from .model import UNet

# import the all necessary functions from submodules and define them in the package namespace

from .augmentation import augment_images as augment
from .training import train_nanopore_detector as train
from .detector import predict_nanopores as detect
from .analyser import process_single_mask as analyze
from .finetuning import main as finetune

# Define what gets imported when using 'from porodet import
__all__ = [
    'UNet', 
    'augment', 
    'train', 
    'detect', 
    'analyze', 
    'finetune'
]

# Package metadata
__version__ = '0.1.0'
__author__ = 'Deep7285', 'rajatnama'