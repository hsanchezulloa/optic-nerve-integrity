# Import the necessary libraries.

# Stdlib
import os
import subprocess

# Core scientific stack
import numpy as np
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Medical imaging
import nibabel as nib
import monai

# Pillow
from PIL import Image
from PIL import __version__ as PILLOW_VERSION  # Pillow version if you need it

# TorchVision transforms
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Metrics & utilities
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

import random

# 2.1.2 Read the files and store them in the dir_3D_list dictionary.

def extract_identifier(file_name):
    return file_name.split('_')[0]

##

# 2.1.3 Create slices along the images (axial plane).

def process_3d_to_2d(dir3D_list):
    """
    Process 3D medical images into 2D slices.
    Keeps only non-black slices and stores them in a dictionary.
    
    Args:
        dir3D_list (dict): Dictionary of 3D images (e.g., nibabel objects).
        
    Returns:
        dict: Dictionary where keys are patient IDs and values are dicts with:
              - 'image': list of 2D slices
              - 'slices': list of slice indices
    """
    # Initialize dictionaries for images
    dir2D_dir = {}   
    
    # Now process the images (without masks)
    transform_PIL = transforms.ToPILImage() 
    keys = dir3D_list.keys() 
    print(keys)

    # For each 3D image (without corresponding mask)
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

            image_slice = np.transpose(image_slice, [1,2,0])
            
            # Check if the image slice is completely black (i.e., all values are 0)
            if torch.all(image_slice == 0):  # Image is black
                continue  # Skip this slice if it's entirely black
            
            # If the image slice is not black, append the image slice
            dir2D_list.append(image_slice)
            cont_list.append(cont)
        
        # Store the 2D slices for the current patient
        dir2D_dir[n] = {'image': dir2D_list, 'slices': cont_list}
        print(f"{n} has {cont} slices, {len(dir2D_dir[n]['image'])} valid image slices.")

    # Final output with the sliced 2D images
    print(f"Processed {len(dir2D_dir)} patients' 2D slices.")
    
    return dir2D_dir

##

# 2.1.4 Creating the different patches

def extract_defined_patches(image, patch_size, black_threshold, num_slice):
    patches_list_image = []
    patches_list_coordinates = []
      # Ensure image is 2D
    if len(image.shape) > 2: 
        image = image.squeeze()  # Removes any singleton dimension
    
    h, w = image.shape  # image shape is [H, W] (for 2D slices)
    patch_h, patch_w = patch_size
    
    total_pixels = patch_h * patch_w

    # Extract the patch
    for i in range(0, h - patch_h + 1, patch_h):
        for j in range(0, w - patch_w + 1, patch_w):
            image_patch = image[i:i+patch_h, j:j+patch_w]

            # Calculate percentage of black pixels
            num_black_pixels = torch.sum(image_patch == 0).item()  # count black
            black_percentage = num_black_pixels / total_pixels

            # If all pixels in the patch are black, skip this patch
            if black_percentage > black_threshold:
                continue
            # if torch.all(image_patch == 0):  # Check if all pixels in the patch are black
            #     continue  # Skip this patch if it's all black
            
            patches_list_image.append(image_patch)

            if i==0 and j==64:
                patches_list_coordinates.append([32,208,num_slice-1])
            elif i==0 and j==96:
                patches_list_coordinates.append([64,208,num_slice-1])
            elif i==0 and j==128:
                patches_list_coordinates.append([96,208,num_slice-1])

            elif i==32 and j==64:
                patches_list_coordinates.append([32,176,num_slice-1])
            elif i==32 and j==96:
                patches_list_coordinates.append([64,176,num_slice-1])
            elif i==32 and j==128:
                patches_list_coordinates.append([96,144,num_slice-1])

            elif i==64 and j==32:
                patches_list_coordinates.append([64,240,num_slice-1])
            elif i==64 and j==64:
                patches_list_coordinates.append([64,176,num_slice-1])
            elif i==64 and j==96:
                patches_list_coordinates.append([64,144,num_slice-1])
            elif i==64 and j==128:
                patches_list_coordinates.append([64,112,num_slice-1])
            elif i==64 and j==160:
                patches_list_coordinates.append([64,80,num_slice-1])
            elif i==64 and j==192:
                patches_list_coordinates.append([64,48,num_slice-1])
            elif i==64 and j==224:
                patches_list_coordinates.append([64,16,num_slice-1])

            else:
                patches_list_coordinates.append([64, j + patch_w // 2, num_slice - 1])
    
    return patches_list_image, patches_list_coordinates

##

# 2.1.5 Save the patches in the corresponding folder

def save_patches_to_nifti(patches_dir2D, patches_dir2D_coordinates, base_output_dir):
    """
    Save 2D image patches as NIfTI files in a structured directory.

    Args:
        patches_dir2D (dict): Dictionary mapping patient_id -> {'image': list of patches}.
        patches_dir2D_coordinates (dict): Dictionary mapping patient_id -> list of coordinates.
        base_output_dir (str): Base directory where the patches will be saved.

    Returns:
        None
    """
    # Ensure base directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    # Iterate over all patients
    for patient_id, patch_dict in patches_dir2D.items():
        image_patches = patch_dict['image']
        coordinate_patches = patches_dir2D_coordinates[patient_id]

        # Shorten patient_id
        patient_id_short, side = patient_id.rsplit("-", 1)

        # Create patient-specific output directory
        patient_image_dir = os.path.join(base_output_dir, side, patient_id_short)
        os.makedirs(patient_image_dir, exist_ok=True)

        # Save each patch
        for cont, image_patch in enumerate(image_patches):
            # Convert to numpy if needed
            patch_image_np = image_patch.cpu().numpy() if isinstance(image_patch, torch.Tensor) else image_patch

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

        print(f"Saved {len(image_patches)} patches for {patient_id_short} in /{side}/{patient_id_short}")

    print("All patches have been saved successfully.")

##

#2.1.6 OPTIONAL: Reconstruct 3D volume from the patches

def reconstruct_3d_volume(
    base_patch_dir,
    patient_id_short,
    side,
    output_base_dir,
    reference_image_path,
    volume_shape=(176, 240, 165),
    patch_size=32,
):
    """
    Reconstruct a 3D volume from saved 2D patches and optionally copy geometry
    from a reference image (via FSL's fslcpgeom).

    Args:
        base_patch_dir (str): Root folder containing patches (organized as <base>/<side>/<patient_id_short>/).
        patient_id_short (str): Patient identifier prefix used in your patch directories (e.g., 'sub115' or 'subject115').
        side (str): Hemisphere/side string (e.g., 'right' or 'left').
        output_base_dir (str): Base directory to save reconstructed 3D images, a subfolder per side will be created.
        reference_image_path (str): Full path to the source NIfTI whose geometry you want to copy (fslcpgeom source).
        volume_shape (tuple[int,int,int], optional): (H, W, D). Defaults to (176, 240, 165).
        patch_size (int, optional): Patch side length. Defaults to 32.

    Returns:
        str: Path to the saved reconstructed NIfTI file.
    """
    VOLUME_HEIGHT, VOLUME_WIDTH, VOLUME_DEPTH = volume_shape
    PATCH_SIZE = patch_size

    # Paths for patches and output
    image_patch_dir = os.path.join(base_patch_dir, side, patient_id_short)
    reconstructed_image_volume = np.zeros((VOLUME_HEIGHT, VOLUME_WIDTH, VOLUME_DEPTH), dtype=np.float32)

    # All patch files
    image_patch_files = sorted(os.listdir(image_patch_dir))

    # Reconstruct volume
    for filename in image_patch_files:
        patch_path = os.path.join(image_patch_dir, filename)

        # Load NIfTI patch
        patch_img = nib.load(patch_path)

        # Extract coordinates from filename: expected "patch_x_y_slice.nii.gz"
        parts = filename.replace('patch_', '').replace('.nii.gz', '').split('_')
        x, y, slice_num = map(int, parts)

        # Get patch data
        patch_data = patch_img.get_fdata()

        # Ensure (H, W, 1)
        if patch_data.ndim == 2:
            patch_data = patch_data[..., np.newaxis]
        elif patch_data.ndim == 3 and patch_data.shape[-1] != 1:
            raise ValueError(f"Patch has unexpected shape {patch_data.shape}")

        assert patch_data.shape == (PATCH_SIZE, PATCH_SIZE, 1), f"Unexpected patch size for {filename}"

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
    # reference_image_path should be like: ".../images/<side>/<patient_id_short>_<side>_T1.nii.gz"
    subprocess.run(['fslcpgeom', reference_image_path, image_output_path], check=False)

    return image_output_path

##
# 2.1.7 Normalize the patches

def normalize_image_max(input_image, output_image):
    # Get max intensity using fslstats
    result = subprocess.run(
        ['fslstats', input_image, '-R'],
        stdout=subprocess.PIPE,
        text=True
    )
    min_val, max_val = map(float, result.stdout.split())

    # Run fslmaths to divide by max intensity
    subprocess.run(['fslmaths', input_image, '-div', str(max_val), output_image])

def normalize_all_patches(input_folder, output_folder):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files ending with .nii or .nii.gz in input_folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder,filename)
            print(f'Normalizing {input_path} -> {output_path}')
            normalize_image_max(input_path, output_path)


##############################################################################################################################################################################################
##############################################################################################################################################################################################
##
# 2.2.1 Load the patches

class SegmentationDataset(Dataset):
    def __init__(self, image_dirs, transform=None):
        """
        Args:
            image_dirs (list of str): List of directories containing image files.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dirs = image_dirs
        self.subjects = []

        # Build the list of images for each subject
        for image_dir in image_dirs:
            image_filenames = sorted(os.listdir(image_dir))
            self.subjects.append(image_filenames)

        self.transform = transform

    def __len__(self):
        """Returns the total number of patches across all subjects."""
        return sum(len(subject) for subject in self.subjects)

    def __getitem__(self, idx):
        """Returns an image."""
        total_patches = 0
        for subject_idx, image_filenames in enumerate(self.subjects):
            num_patches = len(image_filenames)
            if total_patches + num_patches > idx:
                local_idx = idx - total_patches
                image_path = os.path.join(self.image_dirs[subject_idx], image_filenames[local_idx])
                break
            total_patches += num_patches

        # Load the image
        image = nib.load(image_path).get_fdata()  # Load image data

        # Handle 3D or 4D images
        if len(image.shape) == 3:
            middle_slice = image.shape[2] // 2
            image = image[:, :, middle_slice]  # Take middle slice [H, W]
        elif len(image.shape) == 4:
            middle_slice = image.shape[2] // 2
            image = image[:, :, middle_slice, 0]  # Take middle slice and first channel

        # Convert to torch tensor and add channel dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Apply transformations if provided (e.g., resize, normalization)
        if self.transform:
            image = self.transform(image)

        return image

##
# 2.2.2 Load the segmentation algorithm structure (UNet)

def create_monai_unet():
    model = monai.networks.nets.UNet(
    spatial_dims=2, # as it is a 2D image
    in_channels=1, # as it is a grayscale image
    out_channels=1, # binary segmentation 
    channels=(16, 32, 64, 128), # number of channels for each block in the network.
    strides=(2,2,2), # length of this list must be `len(channels) - 1`. Typically, you use strides of 2 for downsampling.
    kernel_size = 3, # typically 3 or 5.
    up_kernel_size = 3, # upsize kernel
    num_res_units = 0, # residual units, deeper levels of connection.
    act = 'ReLU', # activation function.
    norm = 'batch', # batch normalitzation.
    dropout = 0.2, # number of dropouts.
    #adn_ordering = 'NDA' # normalitzation, dropout and activation order.
    )
    return model

##
# 2.2.3 Visualization of some testing

def load_weights(model, model_weights_path):
    ckpt = torch.load(model_weights_path, map_location="cpu")
    state_dict = ckpt
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "net", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
    model.load_state_dict(state_dict, strict=True)

def evaluate_model(model, test_loader, criterion=None, model_weights_path=None, threshold=0.5, device = None):
    """
    Works with datasets that return:
      - only images  -> batch is Tensor
      - (images, masks) -> batch is (Tensor, Tensor)
      - dicts with keys 'image' / 'mask'
    If masks are absent, loss/metrics are skipped.
    """
    if model_weights_path:
        load_weights(model, model_weights_path)
        print(f"Loaded weights: {model_weights_path}")

    model.to(device).eval()

    total_loss = 0.0
    total_batches = 0
    total_acc = total_prec = total_rec = total_dice = total_jacc = 0.0
    have_masks = False  # will flip to True if we detect masks in the first batch

    with torch.inference_mode():
        for batch in test_loader:
            # --- unpack batch robustly ---
            images = masks = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 1:
                    images = batch[0]
                elif len(batch) >= 2:
                    images, masks = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get('image') or batch.get('img') or batch.get('x')
                masks  = batch.get('mask')  or batch.get('label') or batch.get('y')
            else:
                images = batch  # plain Tensor

            if images is None:
                raise ValueError("Could not find images in batch.")

            images = images.to(device, non_blocking=True)

            logits = model(images)                  # (B,1,H,W) or similar
            probs  = torch.sigmoid(logits)         # [0,1]
            preds  = (probs >= threshold).float()  # binary

            # Only compute loss/metrics if masks exist
            if masks is not None:
                have_masks = True
                masks = masks.to(device, non_blocking=True).float()

                if criterion is not None:
                    loss = criterion(logits, masks)
                    total_loss += float(loss.item())

                # Replace with your own metric function; expect (preds, masks)
                acc, prec, rec, dice, jacc = compute_metrics(preds, masks)
                total_acc   += acc
                total_prec  += prec
                total_rec   += rec
                total_dice  += dice
                total_jacc  += jacc

            total_batches += 1

    # Summaries
    if total_batches == 0:
        raise RuntimeError("Empty test_loader.")

    results = {"batches": total_batches}

    if have_masks:
        results.update({
            "loss":   (total_loss / total_batches) if criterion is not None else None,
            "acc":    total_acc  / total_batches,
            "prec":   total_prec / total_batches,
            "rec":    total_rec  / total_batches,
            "dice":   total_dice / total_batches,
            "jacc":   total_jacc / total_batches,
        })
        print("With masks -> metrics:")
        print("Loss: {loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | "
              "Rec: {rec:.4f} | Dice: {dice:.4f} | Jacc: {jacc:.4f}".format(**{k:(v if v is not None else 0.0) for k,v in results.items()}))
    else:
        print(f"Ran inference on {total_batches} batches (images only). No loss/metrics computed.")

    return results

##

def visualize_segmentation_predictions(model, test_dataset, num_images=4):
    """
    Visualize input images and model predictions (no ground truth).
    
    Parameters:
    - model: Trained U-Net model
    - test_dataset: Original test dataset (not the DataLoader)
    - num_images: Number of images to visualize
    """
    # Set model to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Randomly select indices to visualize
    indices = random.sample(range(len(test_dataset)), num_images)
    print(f"Selected indices: {indices}")
    
    # Prepare the figure
    fig, axs = plt.subplots(num_images, 2, figsize=(15, 4*num_images))
    fig.suptitle('Segmentation Results: Input Image | Predicted Mask', fontsize=16)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image from dataset (only image, no mask needed)
            image = test_dataset[idx]  # Get the image from the dataset
            
            # Add batch dimension (model expects batch size as the first dimension)
            image = image.unsqueeze(0).to(device)
            
            # Get model predictions
            output = model(image)
            
            # Convert to probability map (sigmoid)
            pred_mask = torch.sigmoid(output)
            pred_mask = (pred_mask > 0.5).float()  # Threshold at 0.5 to get binary mask
            
            # Move tensors to CPU for visualization and convert to numpy
            image = image.cpu().squeeze(0).squeeze(0).numpy()  # Remove batch and channel dimensions
            pred_mask = pred_mask.cpu().squeeze(0).squeeze(0).numpy()
            
            # Plot the original image
            axs[i, 0].imshow(image, cmap='gray')
            axs[i, 0].set_title(f'Input Image {idx}')
            axs[i, 0].axis('off')
            
            # Plot the predicted mask
            axs[i, 1].imshow(pred_mask, cmap='gray')
            axs[i, 1].set_title(f'Predicted Mask {idx}')
            axs[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

##
# 2.2.4 Make and save the predictions

def predict_and_save_masks(
    model,
    test_dataset,
    test_image_dirs,
    output_base_dir,
    threshold=0.5,
    device=None,
):
    """
    Run inference over patch directories, save predicted masks as NIfTI files.

    Assumes:
      - `test_dataset[idx]` returns a single image tensor (no mask).
      - The dataset indexing aligns with the concatenation order of files
        in `test_image_dirs` (same assumption as your original loop).
      - Destination path is constructed as <output_base_dir>/<side>/<case>/filename.nii.gz
        where <side> and <case> are the last two components of each input path.

    Args:
        model: PyTorch model (binary segmentation).
        test_dataset: Dataset that returns image tensors by index.
        test_image_dirs (list[str]): Directories containing the input patch files.
        output_base_dir (str): Base directory where predicted masks will be saved.
        threshold (float): Sigmoid threshold for binarizing predictions.
        device (torch.device | None): If None, auto-select CUDA if available.

    Returns:
        list[str]: Paths to saved NIfTI mask files.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)

    saved_paths = []
    final_index = 0  # cumulative index offset across directories

    for paths in test_image_dirs:
        # Sorted list of filenames that are files
        files = sorted([f for f in os.listdir(paths) if os.path.isfile(os.path.join(paths, f))])
        num_images = len(files)
        final_index += num_images

        if num_images == 0:
            print(f"Warning: No files found in directory {paths}. Skipping...")
            continue

        # Indices for this directory within the concatenated dataset
        start_idx = final_index - num_images
        indices = list(np.arange(start_idx, final_index))

        with torch.no_grad():
            for i, idx in enumerate(indices):
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
                    raise ValueError(f"Cannot infer side/case from path: {paths}")
                side = path_parts[-2]
                case = path_parts[-1]

                # Destination directory
                new_directory = os.path.join(output_base_dir, side, case)
                os.makedirs(new_directory, exist_ok=True)

                # Save with the same filename as the source patch
                filename = files[i]
                mask_nifti_path = os.path.join(new_directory, filename)
                nib.save(nifti_image, mask_nifti_path)

                print(f"Saved predicted mask for {filename} at {mask_nifti_path}")
                saved_paths.append(mask_nifti_path)

    return saved_paths


##############################################################################################################################################################################################
##############################################################################################################################################################################################
##

# 2.3.1 To make and save the 3D predictions