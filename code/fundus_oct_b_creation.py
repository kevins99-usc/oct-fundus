import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_oct_volume(file_path):
    """
    Load OCT volume data. This function should be adapted based on your specific file format.
    Common formats include .npy, .mat, DICOM series, or custom binary formats.
    
    Args:
        file_path: Path to the OCT volume file
    
    Returns:
        oct_volume: 3D numpy array with shape (depth, height, width)
    """
    # Example for .npy files - adjust based on your actual file format
    if file_path.endswith('.npy'):
        oct_volume = np.load(file_path)
    else:
        # This is a placeholder - you would need to implement specific loaders
        # for your actual file format (DICOM, proprietary formats, etc.)
        raise NotImplementedError(f"Loading {os.path.splitext(file_path)[1]} files not implemented")
    
    # Ensure correct dimensions (depth, height, width)
    if oct_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {oct_volume.shape}")
    
    return oct_volume

def generate_fundus_projection(oct_volume, method='sum'):
    """
    Generate a fundus projection from OCT volume.
    
    Args:
        oct_volume: 3D numpy array with shape (depth, height, width)
        method: Projection method - 'sum', 'max', 'mean'
    
    Returns:
        fundus_image: 2D numpy array with shape (height, width)
    """
    if method == 'sum':
        fundus = np.sum(oct_volume, axis=0)
    elif method == 'max':
        fundus = np.max(oct_volume, axis=0)
    elif method == 'mean':
        fundus = np.mean(oct_volume, axis=0)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    # Normalize to 0-255 for visualization
    fundus = cv2.normalize(fundus, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Optional: Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    fundus = clahe.apply(fundus)
    
    return fundus

def generate_oct_b_scan(oct_volume, scan_index=None, direction='horizontal'):
    """
    Extract a B-scan from the OCT volume.
    
    Args:
        oct_volume: 3D numpy array with shape (depth, height, width)
        scan_index: Index for the B-scan. If None, uses middle of the volume.
        direction: 'horizontal' (default) or 'vertical'
    
    Returns:
        b_scan: 2D numpy array representing the B-scan
    """
    depth, height, width = oct_volume.shape
    
    if direction == 'horizontal':
        if scan_index is None:
            scan_index = height // 2
        b_scan = oct_volume[:, scan_index, :]
    elif direction == 'vertical':
        if scan_index is None:
            scan_index = width // 2
        b_scan = oct_volume[:, :, scan_index]
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    # Normalize to 0-255 for visualization
    b_scan = cv2.normalize(b_scan, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Optional: Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b_scan = clahe.apply(b_scan)
    
    return b_scan

def generate_oct_b_scan_series(oct_volume, direction='horizontal', step=1):
    """
    Generate a series of B-scans from the OCT volume.
    
    Args:
        oct_volume: 3D numpy array with shape (depth, height, width)
        direction: 'horizontal' (default) or 'vertical'
        step: Step size between B-scans
    
    Returns:
        b_scans: List of B-scan images
    """
    depth, height, width = oct_volume.shape
    b_scans = []
    
    if direction == 'horizontal':
        for i in range(0, height, step):
            b_scan = generate_oct_b_scan(oct_volume, scan_index=i, direction='horizontal')
            b_scans.append(b_scan)
    elif direction == 'vertical':
        for i in range(0, width, step):
            b_scan = generate_oct_b_scan(oct_volume, scan_index=i, direction='vertical')
            b_scans.append(b_scan)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return b_scans

def apply_brightness_contrast(image, brightness=0, contrast=0):
    """
    Apply brightness and contrast adjustments to an image.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
    
    Returns:
        Adjusted image
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image

def visualize_results(fundus_image, b_scans, output_dir=None):
    """
    Visualize and optionally save the fundus image and B-scans.
    
    Args:
        fundus_image: Fundus projection image
        b_scans: List of B-scan images or a single B-scan
        output_dir: Directory to save images (optional)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Show fundus image
    plt.subplot(1, 2, 1)
    plt.imshow(fundus_image, cmap='gray')
    plt.title('Fundus Projection')
    plt.axis('off')
    
    # Show middle B-scan or first B-scan from the list
    plt.subplot(1, 2, 2)
    if isinstance(b_scans, list):
        b_scan_to_show = b_scans[len(b_scans)//2]  # Show middle scan
    else:
        b_scan_to_show = b_scans
    
    plt.imshow(b_scan_to_show, cmap='gray')
    plt.title('B-scan')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save fundus
        cv2.imwrite(str(output_dir / 'fundus.png'), fundus_image)
        
        # Save B-scans
        if isinstance(b_scans, list):
            for i, b_scan in enumerate(b_scans):
                cv2.imwrite(str(output_dir / f'b_scan_{i:03d}.png'), b_scan)
        else:
            cv2.imwrite(str(output_dir / 'b_scan.png'), b_scans)
    
    plt.show()

def process_oct_volume(file_path, output_dir=None):
    """
    Main function to process an OCT volume and generate fundus and B-scan images.
    
    Args:
        file_path: Path to the OCT volume file
        output_dir: Directory to save output images (optional)
    """
    # Load the OCT volume
    print(f"Loading OCT volume from {file_path}...")
    oct_volume = load_oct_volume(file_path)
    print(f"OCT volume loaded with shape: {oct_volume.shape}")
    
    # Generate fundus projection
    print("Generating fundus projection...")
    fundus_image = generate_fundus_projection(oct_volume, method='sum')
    
    # Generate horizontal B-scan from the middle of the volume
    print("Generating horizontal B-scan...")
    horizontal_b_scan = generate_oct_b_scan(oct_volume, direction='horizontal')
    
    # Generate vertical B-scan from the middle of the volume
    print("Generating vertical B-scan...")
    vertical_b_scan = generate_oct_b_scan(oct_volume, direction='vertical')
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(fundus_image, horizontal_b_scan, output_dir)
    
    # Save all results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save horizontal B-scan
        cv2.imwrite(str(output_dir / 'horizontal_b_scan.png'), horizontal_b_scan)
        
        # Save vertical B-scan
        cv2.imwrite(str(output_dir / 'vertical_b_scan.png'), vertical_b_scan)
        
        print(f"Results saved to {output_dir}")
    
    return fundus_image, horizontal_b_scan, vertical_b_scan

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process OCT volume to generate fundus and B-scan images')
    parser.add_argument('file_path', type=str, help='Path to the OCT volume file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output images')
    parser.add_argument('--b_scan_series', action='store_true', help='Generate series of B-scans')
    parser.add_argument('--step', type=int, default=5, help='Step size for B-scan series')
    args = parser.parse_args()
    
    # Load the OCT volume
    oct_volume = load_oct_volume(args.file_path)
    
    # Generate fundus projection
    fundus_image = generate_fundus_projection(oct_volume)
    
    if args.b_scan_series:
        # Generate series of B-scans
        horizontal_b_scans = generate_oct_b_scan_series(oct_volume, direction='horizontal', step=args.step)
        output_dir = args.output_dir or 'output'
        visualize_results(fundus_image, horizontal_b_scans, output_dir)
    else:
        # Generate single B-scans
        horizontal_b_scan = generate_oct_b_scan(oct_volume, direction='horizontal')
        vertical_b_scan = generate_oct_b_scan(oct_volume, direction='vertical')
        
        output_dir = args.output_dir or 'output'
        visualize_results(fundus_image, horizontal_b_scan, output_dir)