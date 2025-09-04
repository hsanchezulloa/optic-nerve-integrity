"""
Medical Image Patch-based Segmentation Pipeline

This module provides functions for processing 3D medical images into 2D patches,
training/evaluating segmentation models, and reconstructing 3D volumes from predictions.

Key Features:
- 3D to 2D slice conversion with patch extraction
- U-Net based segmentation using MONAI
- 3D volume reconstruction from predicted patches
- Support for NIfTI format medical images

Dependencies:
- PyTorch
- MONAI
- NiBabel
- NumPy
- Matplotlib
- scikit-learn
- FSL (for geometry copying)
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nibabel as nib
import monai
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_identifier(file_name: str) -> str:
    """
    Extract patient identifier from filename.
    
    Args:
        file_name: Filename to extract identifier from
        
    Returns:
        Patient identifier (first part before underscore)
    """
    return file_name.split('_')[0]


def process_3d_to_2d(dir3D_list: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
    """
    Process 3D medical images into 2D slices.
    Keeps only non-black slices and stores them in a dictionary.
    
    Args:
        dir3D_list: Dictionary of 3D images (e.g., nibabel objects).
        
    Returns:
        Dictionary where keys are patient IDs and values are dicts with:
        - 'image': list of 2D slices
        - 'slices': list of slice indices
    """
    dir2D_dir = {}   
    keys = dir3D_list.keys() 
    logger.info(f"Processing {len(keys)} 3D images: {list(keys)}")

    for n in keys:
        # Get the 3D data for the image
        image_data = dir3D_list[n].get_fdata()  
        dir2D_list = []  # List to store the 2D slices for this patient
        cont_list = []  # List to store the slice indices
        
        cont = 0  # Counter for the number of slices
        
        # Process each slice in the z-direction
        for i in range(image_data.shape[2]):
            cont += 1  # Increment counter for slices
            
            # Extract the image slice (2D) and convert to tensor
            image_slice = torch.tensor(image_data[:, :, i], dtype=torch.float32)
            
            # Add a channel dimension to the 2D slice (to make it [C, H, W])
            image_slice = image_slice.unsqueeze(0)  # Shape becomes [1, H, W]
            
            # Rotate the image slice by 90 degrees (optional based on your need)
            image_slice = TF.rotate(image_slice, 90)
            image_slice = np.transpose(image_slice, [1, 2, 0])
            
            # Check if the image slice is completely black (i.e., all values are 0)
            if torch.all(image_slice == 0):  # Image is black
                continue  # Skip this slice if it's entirely black
            
            # If the image slice is not black, append the image slice
            dir2D_list.append(image_slice)
            cont_list.append(cont)
        
        # Store the 2D slices for the current patient
        dir2D_dir[n] = {'image': dir2D_list, 'slices': cont_list}
        logger.info(f"{n} has {cont} slices, {len(dir2D_dir[n]['image'])} valid image slices.")

    logger.info(f"Processed {len(dir2D_dir)} patients' 2D slices.")
    return dir2D_dir


def extract_defined_patches(image: torch.Tensor, patch_size: Tuple[int, int], 
                          black_threshold: float, num_slice: int) -> Tuple[List, List]:
    """
    Extract patches from a 2D image with specific coordinate mapping.
    
    Args:
        image: 2D image tensor
        patch_size: Tuple of (height, width) for patch size
        black_threshold: Maximum fraction of black pixels allowed in patch
        num_slice: Slice number for coordinate tracking
        
    Returns:
        Tuple of (patches_list, coordinates_list)
    """
    patches_list_image = []
    patches_list_coordinates = []
    
    # Ensure image is 2D
    if len(image.shape) > 2: 
        image = image.squeeze()  # Removes any singleton dimension
    
    h, w = image.shape  # image shape is [H, W] (for 2D slices)
    patch_h, patch_w = patch_size
    total_pixels = patch_h * patch_w

    # Extract patches
    for i in range(0, h - patch_h + 1, patch_h):
        for j in range(0, w - patch_w + 1, patch_w):
            image_patch = image[i:i+patch_h, j:j+patch_w]

            # Calculate percentage of black pixels
            num_black_pixels = torch.sum(image_patch == 0).item()
            black_percentage = num_black_pixels / total_pixels

            # Skip patches with too many black pixels
            if black_percentage > black_threshold:
                continue
            
            patches_list_image.append(image_patch)

            # Specific coordinate mapping (this seems to be brain region specific)
            # You may want to make this more configurable
            coordinate_map = {
                (0, 64): [32, 208, num_slice-1],
                (0, 96): [64, 208, num_slice-1],
                (0, 128): [96, 208, num_slice-1],
                (32, 64): [32, 176, num_slice-1],
                (32, 96): [64, 176, num_slice-1],
                (32, 128): [96, 144, num_slice-1],
                (64, 32): [64, 240, num_slice-1],
                (64, 64): [64, 176, num_slice-1],
                (64, 96): [64, 144, num_slice-1],
                (64, 128): [64, 112, num_slice-1],
                (64, 160): [64, 80, num_slice-1],
                (64, 192): [64, 48, num_slice-1],
                (64, 224): [64, 16, num_slice-1],
            }
            
            coord = coordinate_map.get((i, j), [64, j + patch_w // 2, num_slice - 1])
            patches_list_coordinates.append(coord)
    
    return patches_list_image, patches_list_coordinates


def save_patches_to_nifti(patches_dir2D: Dict[str, Dict], 
                         patches_dir2D_coordinates: Dict[str, List], 
                         base_output_dir: str) -> None:
    """
    Save 2D image patches as NIfTI files in a structured directory.

    Args:
        patches_dir2D: Dictionary mapping patient_id -> {'image': list of patches}.
        patches_dir2D_coordinates: Dictionary mapping patient_id -> list of coordinates.
        base_output_dir: Base directory where the patches will be saved.
    """
    os.makedirs(base_output_dir, exist_ok=True)

    for patient_id, patch_dict in patches_dir2D.items():
        image_patches = patch_dict['image']
        coordinate_patches = patches_dir2D_coordinates[patient_id]

        # Split patient_id to get side information
        if "-" in patient_id:
            patient_id_short, side = patient_id.rsplit("-", 1)
        else:
            # Fallback if format is different
            patient_id_short = patient_id
            side = "unknown"
            logger.warning(f"Could not parse side from patient_id: {patient_id}")

        # Create patient-specific output directory
        patient_image_dir = os.path.join(base_output_dir, side, patient_id_short)
        os.makedirs(patient_image_dir, exist_ok=True)

        # Save each patch
        for cont, image_patch in enumerate(image_patches):
            # Convert to numpy if needed
            if isinstance(image_patch, torch.Tensor):
                patch_image_np = image_patch.cpu().numpy()
            else:
                patch_image_np = image_patch

            # Flip and rotate
            patch_image_np = np.flip(patch_image_np, axis=0)
            patch_image_np = np.rot90(patch_image_np, k=1)

            # Create a NIfTI image
            nifti_image = nib.Nifti1Image(patch_image_np, affine=np.eye(4))

            # File name based on coordinates
            name_patch = coordinate_patches[cont]
            image_nifti_path = os.path.join(
                patient_image_dir,
                f"patch_{name_patch[0]}_{name_patch[1]}_{name_patch[2]}.nii.gz"
            )

            # Save
            nib.save(nifti_image, image_nifti_path)

        logger.info(f"Saved {len(image_patches)} patches for {patient_id_short} in /{side}/{patient_id_short}")

    logger.info("All patches have been saved successfully.")


def reconstruct_3d_volume(base_patch_dir: str, patient_id_short: str, side: str,
                         output_base_dir: str, reference_image_path: str,
                         volume_shape: Tuple[int, int, int] = (176, 240, 165),
                         patch_size: int = 32) -> str:
    """
    Reconstruct a 3D volume from saved 2D patches and optionally copy geometry
    from a reference image (via FSL's fslcpgeom).

    Args:
        base_patch_dir: Root folder containing patches (organized as <base>/<side>/<patient_id_short>/).
        patient_id_short: Patient identifier prefix used in patch directories.
        side: Hemisphere/side string (e.g., 'right' or 'left').
        output_base_dir: Base directory to save reconstructed 3D images.
        reference_image_path: Full path to the source NIfTI whose geometry you want to copy.
        volume_shape: (H, W, D) dimensions of the output volume.
        patch_size: Patch side length.

    Returns:
        Path to the saved reconstructed NIfTI file.
    """
    VOLUME_HEIGHT, VOLUME_WIDTH, VOLUME_DEPTH = volume_shape
    PATCH_SIZE = patch_size

    # Paths for patches and output
    image_patch_dir = os.path.join(base_patch_dir, side, patient_id_short)
    if not os.path.exists(image_patch_dir):
        raise FileNotFoundError(f"Patch directory not found: {image_patch_dir}")
    
    reconstructed_image_volume = np.zeros((VOLUME_HEIGHT, VOLUME_WIDTH, VOLUME_DEPTH), dtype=np.float32)

    # All patch files
    image_patch_files = sorted(os.listdir(image_patch_dir))
    if not image_patch_files:
        raise ValueError(f"No patch files found in {image_patch_dir}")

    # Reconstruct volume
    for filename in image_patch_files:
        patch_path = os.path.join(image_patch_dir, filename)

        # Load NIfTI patch
        patch_img = nib.load(patch_path)

        # Extract coordinates from filename: expected "patch_x_y_slice.nii.gz"
        try:
            parts = filename.replace('patch_', '').replace('.nii.gz', '').split('_')
            x, y, slice_num = map(int, parts)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse coordinates from filename {filename}: {e}")
            continue

        # Get patch data
        patch_data = patch_img.get_fdata()

        # Ensure (H, W, 1)
        if patch_data.ndim == 2:
            patch_data = patch_data[..., np.newaxis]
        elif patch_data.ndim == 3 and patch_data.shape[-1] != 1:
            logger.warning(f"Patch has unexpected shape {patch_data.shape} for {filename}")
            continue

        if patch_data.shape != (PATCH_SIZE, PATCH_SIZE, 1):
            logger.warning(f"Unexpected patch size {patch_data.shape} for {filename}")
            continue

        # Check bounds before placing patch
        if (x + PATCH_SIZE > VOLUME_HEIGHT or 
            y < PATCH_SIZE or y > VOLUME_WIDTH or 
            slice_num >= VOLUME_DEPTH):
            logger.warning(f"Patch {filename} coordinates out of bounds, skipping")
            continue

        # Place the (flipped) patch
        reconstructed_image_volume[
            x:x+PATCH_SIZE,
            y-PATCH_SIZE:y,
            slice_num
        ] = np.flip(patch_data[:, :, 0], axis=0)

    # Save reconstructed volume
    reconstructed_image_nifti = nib.Nifti1Image(reconstructed_image_volume, affine=np.eye(4))
    output_dir_image = os.path.join(output_base_dir, side)
    os.makedirs(output_dir_image, exist_ok=True)

    image_output_path = os.path.join(output_dir_image, f'{patient_id_short}_{side}_reconstructed_image.nii.gz')
    nib.save(reconstructed_image_nifti, image_output_path)

    # Copy geometry from the provided reference image (requires FSL)
    if os.path.exists(reference_image_path):
        try:
            result = subprocess.run(['fslcpgeom', reference_image_path, image_output_path], 
                                  check=True, capture_output=True, text=True)
            logger.info(f"Successfully copied geometry from {reference_image_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to copy geometry: {e.stderr}")
        except FileNotFoundError:
            logger.warning("FSL not found. Geometry copying skipped.")
    else:
        logger.warning(f"Reference image not found: {reference_image_path}")

    return image_output_path


def normalize_image_max(input_image: str, output_image: str) -> None:
    """
    Normalize image by dividing by maximum intensity using FSL tools.
    
    Args:
        input_image: Path to input NIfTI image
        output_image: Path to output normalized image
    """
    try:
        # Get max intensity using fslstats
        result = subprocess.run(
            ['fslstats', input_image, '-R'],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        min_val, max_val = map(float, result.stdout.split())

        # Run fslmaths to divide by max intensity
        subprocess.run(['fslmaths', input_image, '-div', str(max_val), output_image], 
                      check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FSL command failed: {e}")
        raise
    except FileNotFoundError:
        logger.error("FSL tools not found. Please install FSL.")
        raise


def normalize_all_patches(input_folder: str, output_folder: str) -> None:
    """
    Normalize all NIfTI patches in a folder.
    
    Args:
        input_folder: Directory containing input patches
        output_folder: Directory to save normalized patches
    """
    os.makedirs(output_folder, exist_ok=True)

    nifti_files = [f for f in os.listdir(input_folder) 
                   if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if not nifti_files:
        logger.warning(f"No NIfTI files found in {input_folder}")
        return

    for filename in nifti_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        logger.info(f'Normalizing {input_path} -> {output_path}')
        normalize_image_max(input_path, output_path)


class SegmentationDataset(Dataset):
    """
    Dataset class for loading image patches for segmentation.
    """
    
    def __init__(self, image_dirs: List[str], transform=None):
        """
        Args:
            image_dirs: List of directories containing image files.
            transform: Optional transform to be applied on an image.
        """
        self.image_dirs = image_dirs
        self.subjects = []

        # Build the list of images for each subject
        for image_dir in image_dirs:
            if not os.path.exists(image_dir):
                logger.warning(f"Directory not found: {image_dir}")
                continue
            image_filenames = sorted(os.listdir(image_dir))
            self.subjects.append(image_filenames)

        self.transform = transform
        logger.info(f"Dataset initialized with {len(self.subjects)} subjects, "
                   f"total patches: {sum(len(s) for s in self.subjects)}")

    def __len__(self) -> int:
        """Returns the total number of patches across all subjects."""
        return sum(len(subject) for subject in self.subjects)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns an image tensor."""
        total_patches = 0
        for subject_idx, image_filenames in enumerate(self.subjects):
            num_patches = len(image_filenames)
            if total_patches + num_patches > idx:
                local_idx = idx - total_patches
                image_path = os.path.join(self.image_dirs[subject_idx], image_filenames[local_idx])
                break
            total_patches += num_patches
        else:
            raise IndexError(f"Index {idx} out of range")

        # Load the image
        try:
            image = nib.load(image_path).get_fdata()
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

        # Handle 3D or 4D images
        if len(image.shape) == 3:
            middle_slice = image.shape[2] // 2
            image = image[:, :, middle_slice]  # Take middle slice [H, W]
        elif len(image.shape) == 4:
            middle_slice = image.shape[2] // 2
            image = image[:, :, middle_slice, 0]  # Take middle slice and first channel

        # Convert to torch tensor and add channel dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image


def create_monai_unet(spatial_dims: int = 2, in_channels: int = 1, out_channels: int = 1,
                     channels: Tuple[int, ...] = (16, 32, 64, 128), 
                     strides: Tuple[int, ...] = (2, 2, 2),
                     kernel_size: int = 3, up_kernel_size: int = 3,
                     num_res_units: int = 0, act: str = 'ReLU',
                     norm: str = 'batch', dropout: float = 0.2) -> nn.Module:
    """
    Create a MONAI U-Net model with configurable parameters.
    
    Args:
        spatial_dims: Number of spatial dimensions (2 for 2D images)
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: Number of channels for each block in the network
        strides: Stride values for downsampling
        kernel_size: Kernel size for convolutions
        up_kernel_size: Kernel size for upsampling
        num_res_units: Number of residual units
        act: Activation function
        norm: Normalization type
        dropout: Dropout rate
        
    Returns:
        MONAI U-Net model
    """
    model = monai.networks.nets.UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        kernel_size=kernel_size,
        up_kernel_size=up_kernel_size,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
    )
    return model


def load_weights(model: nn.Module, model_weights_path: str) -> None:
    """
    Load model weights from a checkpoint file.
    
    Args:
        model: PyTorch model
        model_weights_path: Path to model weights file
    """
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")
    
    ckpt = torch.load(model_weights_path, map_location="cpu")
    state_dict = ckpt
    
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "net", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
    
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded weights from {model_weights_path}")


def compute_metrics(preds: torch.Tensor, masks: torch.Tensor) -> Tuple[float, float, float, float, float]:
    """
    Compute segmentation metrics.
    
    Args:
        preds: Predicted masks (binary)
        masks: Ground truth masks (binary)
        
    Returns:
        Tuple of (accuracy, precision, recall, dice, jaccard)
    """
    preds_flat = preds.view(-1).cpu().numpy()
    masks_flat = masks.view(-1).cpu().numpy()
    
    # Accuracy
    acc = np.mean(preds_flat == masks_flat)
    
    # Precision and Recall
    prec = precision_score(masks_flat, preds_flat, zero_division=0)
    rec = recall_score(masks_flat, preds_flat, zero_division=0)
    
    # Dice coefficient
    intersection = np.sum(preds_flat * masks_flat)
    dice = (2 * intersection) / (np.sum(preds_flat) + np.sum(masks_flat) + 1e-8)
    
    # Jaccard index (IoU)
    union = np.sum(preds_flat) + np.sum(masks_flat) - intersection
    jacc = intersection / (union + 1e-8)
    
    return acc, prec, rec, dice, jacc


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  criterion=None, model_weights_path: Optional[str] = None,
                  threshold: float = 0.5, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        criterion: Loss function (optional)
        model_weights_path: Path to model weights (optional)
        threshold: Threshold for binary predictions
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_weights_path:
        load_weights(model, model_weights_path)

    model.to(device).eval()

    total_loss = 0.0
    total_batches = 0
    total_acc = total_prec = total_rec = total_dice = total_jacc = 0.0
    have_masks = False

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Unpack batch robustly
            images = masks = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 1:
                    images = batch[0]
                elif len(batch) >= 2:
                    images, masks = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get('image') or batch.get('img') or batch.get('x')
                masks = batch.get('mask') or batch.get('label') or batch.get('y')
            else:
                images = batch

            if images is None:
                raise ValueError("Could not find images in batch.")

            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            # Only compute loss/metrics if masks exist
            if masks is not None:
                have_masks = True
                masks = masks.to(device, non_blocking=True).float()

                if criterion is not None:
                    loss = criterion(logits, masks)
                    total_loss += float(loss.item())

                acc, prec, rec, dice, jacc = compute_metrics(preds, masks)
                total_acc += acc
                total_prec += prec
                total_rec += rec
                total_dice += dice
                total_jacc += jacc

            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Empty test_loader.")

    results = {"batches": total_batches}

    if have_masks:
        results.update({
            "loss": (total_loss / total_batches) if criterion is not None else None,
            "acc": total_acc / total_batches,
            "prec": total_prec / total_batches,
            "rec": total_rec / total_batches,
            "dice": total_dice / total_batches,
            "jacc": total_jacc / total_batches,
        })
        logger.info(f"Evaluation complete - Loss: {results.get('loss', 'N/A'):.4f} | "
                   f"Acc: {results['acc']:.4f} | Prec: {results['prec']:.4f} | "
                   f"Rec: {results['rec']:.4f} | Dice: {results['dice']:.4f} | "
                   f"Jacc: {results['jacc']:.4f}")
    else:
        logger.info(f"Ran inference on {total_batches} batches (images only).")

    return results


def visualize_segmentation_predictions(model: nn.Module, test_dataset: Dataset, 
                                     num_images: int = 4, save_path: Optional[str] = None) -> None:
    """
    Visualize input images and model predictions.
    
    Args:
        model: Trained U-Net model
        test_dataset: Test dataset
        num_images: Number of images to visualize
        save_path: Path to save the visualization (optional)
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Randomly select indices to visualize
    indices = random.sample(range(len(test_dataset)), min(num_images, len(test_dataset)))
    logger.info(f"Selected indices for visualization: {indices}")
    
    # Prepare the figure
    fig, axs = plt.subplots(num_images, 2, figsize=(15, 4*num_images))
    if num_images == 1:
        axs = axs.reshape(1, -1)
    fig.suptitle('Segmentation Results: Input Image | Predicted Mask', fontsize=16)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image from dataset
            image = test_dataset[idx]
            
            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)
            
            # Get model predictions
            output = model(image_batch)
            pred_mask = torch.sigmoid(output)
            pred_mask = (pred_mask > 0.5).float()
            
            # Convert to numpy for visualization
            image_np = image.squeeze(0).squeeze(0).cpu().numpy()
            pred_mask_np = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
            
            # Plot the original image
            axs[i, 0].imshow(image_np, cmap='gray')
            axs[i, 0].set_title(f'Input Image {idx}')
            axs[i, 0].axis('off')
            
            # Plot the predicted mask
            axs[i, 1].imshow(pred_mask_np, cmap='gray')
            axs[i, 1].set_title(f'Predicted Mask {idx}')
            axs[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    plt.show()


def predict_and_save_masks(model: nn.Module, test_dataset: Dataset, 
                          test_image_dirs: List[str], output_base_dir: str,
                          threshold: float = 0.5, 
                          device: Optional[torch.device] = None) -> List[str]:
    """
    Run inference over patch directories and save predicted masks as NIfTI files.

    Args:
        model: PyTorch model for binary segmentation
        test_dataset: Dataset that returns image tensors by index
        test_image_dirs: Directories containing the input patch files
        output_base_dir: Base directory where predicted masks will be saved
        threshold: Sigmoid threshold for binarizing predictions
        device: Device to run inference on

    Returns:
        List of paths to saved NIfTI mask files
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)

    saved_paths = []
    final_index = 0  # cumulative index offset across directories

    for paths in test_image_dirs:
        # Sorted list of filenames that are files
        if not os.path.exists(paths):
            logger.warning(f"Directory not found: {paths}. Skipping...")
            continue
            
        files = sorted([f for f in os.listdir(paths) if os.path.isfile(os.path.join(paths, f))])
        num_images = len(files)
        final_index += num_images

        if num_images == 0:
            logger.warning(f"No files found in directory {paths}. Skipping...")
            continue

        # Indices for this directory within the concatenated dataset
        start_idx = final_index - num_images
        indices = list(np.arange(start_idx, final_index))

        with torch.no_grad():
            for i, idx in enumerate(tqdm(indices, desc=f"Processing {os.path.basename(paths)}")):
                try:
                    # Load image from dataset
                    image = test_dataset[idx]          # tensor [C,H,W]
                    image = image.unsqueeze(0).to(device)  # [1,C,H,W]

                    # Forward + threshold
                    output = model(image)
                    pred_mask = torch.sigmoid(output)
                    pred_mask = (pred_mask > threshold).float()

                    # Prepare data for NIfTI (H,W,1)
                    pred_mask_np = pred_mask.cpu().squeeze(0).squeeze(0).numpy()
                    pred_mask_np = np.expand_dims(pred_mask_np, axis=2)

                    # Create NIfTI with identity affine
                    nifti_image = nib.Nifti1Image(pred_mask_np, affine=np.eye(4))

                    # Derive side & case from the input path (last two components)
                    norm_path = os.path.normpath(paths)
                    path_parts = norm_path.split(os.sep)
                    if len(path_parts) < 2:
                        logger.warning(f"Cannot infer side/case from path: {paths}")
                        side = "unknown"
                        case = "unknown"
                    else:
                        side = path_parts[-2]
                        case = path_parts[-1]

                    # Destination directory
                    new_directory = os.path.join(output_base_dir, side, case)
                    os.makedirs(new_directory, exist_ok=True)

                    # Save with the same filename as the source patch
                    filename = files[i]
                    mask_nifti_path = os.path.join(new_directory, filename)
                    nib.save(nifti_image, mask_nifti_path)

                    saved_paths.append(mask_nifti_path)
                    
                except Exception as e:
                    logger.error(f"Error processing patch {i} from {paths}: {e}")
                    continue

        logger.info(f"Processed {len(files)} patches from {paths}")

    logger.info(f"Saved {len(saved_paths)} predicted masks total")
    return saved_paths