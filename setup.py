from setuptools import setup, find_packages

setup(
    name="porodet",
    version="0.1.0",
    author="Deep7285",
    description="Nanoporosity detection in TEM images using U-Net",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python-headless",
        "torch",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "albumentations>=1.3.1",  # Added version constraint for stability
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)