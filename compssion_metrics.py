# Import libraries
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim



# Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # No difference
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Structural Similarity Index (SSIM)
def calculate_ssim(original, compressed):
    ssim_index, _ = ssim(original, compressed, full=True, multichannel=True)
    return ssim_index

# Simulate Multiscale SSIM (MS-SSIM)
def calculate_ms_ssim(original, compressed):
    # Downscale images and compute SSIM at different scales
    scales = [1, 0.5, 0.25]  # Different scales (original, half, quarter resolution)
    ms_ssim_value = 0
    for scale in scales:
        scaled_original = cv2.resize(original, (0, 0), fx=scale, fy=scale)
        scaled_compressed = cv2.resize(compressed, (0, 0), fx=scale, fy=scale)
        ms_ssim_value += calculate_ssim(scaled_original, scaled_compressed)
    return ms_ssim_value / len(scales)

# Bitrate (Bits Per Pixel)
def calculate_bpp(original, compressed_file_path):
    height, width = original.shape[:2]
    num_pixels = height * width
    compressed_size = os.path.getsize(compressed_file_path) * 8  # Convert bytes to bits
    bpp = compressed_size / num_pixels
    return bpp

# Compression Ratio
def calculate_compression_ratio(original_file_path, compressed_file_path):
    original_size = os.path.getsize(original_file_path)
    compressed_size = os.path.getsize(compressed_file_path)
    return original_size / compressed_size
