# Assessing optic nerve integrity in optic neuritis with conventional magnetic resonance imaging
This repository contains the source code for the deep learning model used in the paper "Assessing optic nerve integrity in optic neuritis with conventional magnetic resonance imaging." 

The model provides an automated and objective method for segmenting the optic nerve from routine T1- and T2-weighted brain magnetic resonance imaging (MRI) scans, facilitating the analysis of tissue integrity.

## Overview
The project focuses on developing a robust and reproducible pipeline for assessing optic nerve integrity, which is a crucial biomarker for diagnosing and monitoring multiple sclerosis (MS). The proposed approach automates the segmentation of the optic nerve, a process that is conventionally time-consuming and subject to inter-rater variability.

The core of this work is a 2D U-Net convolutional neural network (CNN). The model was trained on 2D image patches extracted from preprocessed MRI scans, which included a semi-automated pipeline for cropping the optic nerve region and standardizing image orientation.

After segmentation, the optic nerve profiles were extracted from the T1/T2 ratio, which has demonstrated clinical value when non-conventional MRI sequences are unavailable. These profiles were then used to analyze the integrity of the optic nerve in different patient groups, including healthy controls (HC), MS patients without optic neuritis (ON), and MS patients with ON.

## Files and Folders
**`01_data_preprocessing.ipynb`**: Script that extracts fixed-size patches from the region of interest (ROI) around the optic nerve.  
**`02_model_inference.ipynb`**: The core Python script containing the 2D U-Net architecture, training loop, and evaluation functions.  
**`03_reconstruction_analysis.ipynb`**: Script for reconstructing the full optic nerve from the segmented patches.  
**`functions.py`**: A helper file containing reusable functions for the pipeline.  
**`model.pth`**: The trained model weights file.  

## Getting Started
### Prerequisites
To run the code, you will need the following Python libraries. You can install them using pip:

* **`MONAI`** (version 1.3.2) 
* **`Pytorch`** (version 2.4.1) 
* **`NumPy`**, **`pandas`**, **`scikit-learn`**, (for data handling and statistical analysis)

The model was trained on a PC with 12 processors (Intel i7), 128 GB RAM, and a GeForce GTX 1080 Ti with 11 GB graphical memory.
### Usage
This repository contains only the segmentation algorithm and is designed to be integrated into a larger, FSL-based image analysis pipeline.

The **`segmentation_algorithm/`** directory houses the core script for performing segmentation on preprocessed images. The model expects an input of patches that have already been extracted from a previously defined region of interest (ROI). After the algorithm runs, the segmented masks can then be used for subsequent analysis, such as the extraction of optic nerve profiles.