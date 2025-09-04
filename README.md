# Assessing optic nerve integrity in optic neuritis with conventional magnetic resonance imaging
This repository contains the source code for the deep learning model used in the paper "Assessing optic nerve integrity in optic neuritis with conventional magnetic resonance imaging." 

The model provides an automated and objective method for segmenting the optic nerve from routine T1- and T2-weighted brain magnetic resonance imaging (MRI) scans, facilitating the analysis of tissue integrity.

## Overview
The project focuses on developing a robust and reproducible pipeline for assessing optic nerve integrity, which is a crucial biomarker for diagnosing and monitoring multiple sclerosis (MS). The proposed approach automates the segmentation of the optic nerve, a process that is conventionally time-consuming and subject to inter-rater variability.

The core of this work is a 2D U-Net convolutional neural network (CNN). The model was trained on 2D image patches extracted from preprocessed MRI scans, which included a semi-automated pipeline for cropping the optic nerve region and standardizing image orientation.

After segmentation, the optic nerve profiles were extracted from the T1/T2 ratio, which has demonstrated clinical value when non-conventional MRI sequences are unavailable. These profiles were then used to analyze the integrity of the optic nerve in different patient groups, including healthy controls (HC), MS patients without optic neuritis (ON), and MS patients with ON.

## Key Features
* Automated Segmentation: A U-Net CNN is implemented for the automatic, accurate segmentation of the optic nerve from brain MRI data.
* Conventional MRI Utilization: The pipeline is based on standard 3D T1- and T2-weighted MRI, making it highly accessible and suitable for analyzing retrospective cohorts without requiring dedicated sequences.
* Quantitative Analysis: The segmented masks are used to extract longitudinal T1/T2 ratio profiles of the optic nerve. This ratio is used as a surrogate for tissue integrity.
* Clinical Relevance: The approach offers a reproducible tool for assessing optic nerve integrity and monitoring disease progression , with results demonstrating a significant difference in T1/T2 values between affected eyes with lesions and HC, MS without ON, and fellow eyes.

## Files and Folders
**`functions.py/`**: A helper file containing reusable functions for the pipeline.  
**`model.pth/`**: The trained model weights file.  
**`patch_extraction/`**: Script that contains scripts for extracting fixed-size patches from the region of interest (ROI) around the optic nerve.  
**`segmentation_algorithm/`**: The core Python script containing the 2D U-Net architecture, training loop, and evaluation functions.  
**`reconstruction_patches/`**: Scripts for reconstructing the full optic nerve from the segmented patches.  
**`README.md/`**:  This file, providing an overview of the repository.  

## Getting Started
### Prerequisites
To run the code, you will need the following Python libraries. You can install them using pip:

* **`MONAI`** (version 1.3.2) 
* **`Pytorch`** (version 2.4.1) 
* **`NumPy`**, **`pandas`**, **`scikit-learn`**, (for data handling and statistical analysis)

The model was trained on a PC with 12 processors (Intel i7), 128 GB RAM, and a GeForce GTX 1080 Ti with 11 GB graphical memory.
### Usage
This repository contains only the segmentation algorithm and is designed to be integrated into a larger, FSL-based image analysis pipeline.

The **`reconstruction_patches/`** directory houses the core script for performing segmentation on preprocessed images. The model expects an input of patches that have already been extracted from a previously defined region of interest (ROI). After the algorithm runs, the segmented masks can then be used for subsequent analysis, such as the extraction of optic nerve profiles.
