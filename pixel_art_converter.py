#!/usr/bin/env python3
"""
Pixel Art Converter
With CIEDE2000 color distance and advanced preprocessing options.

Usage:
    python pixel_art_converter.py input.png [options]

Options:
    -o, --output        Output filename (default: output.png)
    -p, --pixel-size    Pixel size for downscaling (default: 8)
    -b, --brightness    Brightness adjustment -100 to 200 (default: 0)
    -c, --contrast      Contrast adjustment -100 to 200 (default: 0)
    -s, --saturation    Saturation adjustment -100 to 200 (default: 0)
    --palette           Palette name (default: pico8)
    --no-palette        Disable palette mapping
    --scale-up          Scale output back to original size
    --ciede2000         Use CIEDE2000 color distance
    --preprocess        Preprocessing: auto, histogram, clahe, gamma, match_palette
"""

import argparse
import sys
import math
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Required packages not found. Install with:")
    print("  pip install Pillow numpy")
    sys.exit(1)


# ============================================================================
# Hardcoded Color Palettes
# ============================================================================

PALETTES = {
    "pico8": [
        [0, 0, 0], [29, 43, 83], [126, 37, 83], [0, 135, 81],
        [171, 82, 54], [95, 87, 79], [194, 195, 199], [255, 241, 232],
        [255, 0, 77], [255, 163, 0], [255, 236, 39], [0, 228, 54],
        [41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
    ],
    "lost_century": [
        [209, 177, 135], [199, 123, 88], [174, 93, 64], [121, 68, 74],
        [75, 61, 68], [186, 145, 88], [146, 116, 65], [77, 69, 57],
        [119, 116, 59], [179, 165, 85], [210, 201, 165], [140, 171, 161],
        [75, 114, 110], [87, 72, 82], [132, 120, 117], [171, 155, 142],
    ],
    "sunset8": [
        [255, 255, 120], [255, 214, 71], [255, 194, 71], [255, 169, 54],
        [255, 139, 111], [230, 117, 149], [154, 99, 144], [70, 70, 120],
    ],
    "twilight5": [
        [251, 187, 173], [238, 134, 149], [74, 122, 150],
        [51, 63, 88], [41, 40, 49],
    ],
    "hollow": [
        [15, 15, 27], [86, 90, 117], [198, 183, 190], [250, 251, 246],
    ],
    "gameboy": [
        [15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15],
    ],
    "endesga32": [
        [190, 74, 47], [215, 118, 67], [234, 212, 170], [228, 166, 114],
        [184, 111, 80], [115, 62, 57], [62, 39, 49], [162, 38, 51],
        [228, 59, 68], [247, 118, 34], [254, 174, 52], [254, 231, 97],
        [99, 199, 77], [62, 137, 72], [38, 92, 66], [25, 60, 62],
        [18, 78, 137], [0, 153, 219], [44, 232, 245], [192, 203, 220],
        [139, 155, 180], [90, 105, 136], [58, 68, 102], [38, 43, 68],
        [24, 20, 37], [255, 0, 68], [104, 56, 108], [181, 80, 136],
        [246, 117, 122], [232, 183, 150], [194, 133, 105], [140, 94, 88],
    ],
}


# ============================================================================
# Color Space Conversion Functions
# ============================================================================

def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to XYZ color space (sRGB, D65 illuminant)."""
    rgb_norm = rgb.astype(np.float64) / 255.0
    
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(
        mask,
        ((rgb_norm + 0.055) / 1.055) ** 2.4,
        rgb_norm / 12.92
    )
    
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = np.einsum('...j,ij->...i', rgb_linear, M) * 100.0
    return xyz


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to CIELAB color space (D65 illuminant)."""
    ref_white = np.array([95.047, 100.000, 108.883])
    xyz_norm = xyz / ref_white
    
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    
    mask = xyz_norm > epsilon
    f = np.where(mask, np.cbrt(xyz_norm), (kappa * xyz_norm + 16.0) / 116.0)
    
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to CIELAB."""
    return xyz_to_lab(rgb_to_xyz(rgb))


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to XYZ."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    
    xr = np.where(fx**3 > epsilon, fx**3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * epsilon, ((L + 16.0) / 116.0)**3, L / kappa)
    zr = np.where(fz**3 > epsilon, fz**3, (116.0 * fz - 16.0) / kappa)
    
    ref_white = np.array([95.047, 100.000, 108.883])
    return np.stack([xr * ref_white[0], yr * ref_white[1], zr * ref_white[2]], axis=-1)


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to RGB."""
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    xyz_norm = xyz / 100.0
    rgb_linear = np.einsum('...j,ij->...i', xyz_norm, M_inv)
    
    mask = rgb_linear > 0.0031308
    rgb = np.where(
        mask,
        1.055 * np.power(np.maximum(rgb_linear, 0.0031308), 1/2.4) - 0.055,
        12.92 * rgb_linear
    )
    
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to RGB."""
    return xyz_to_rgb(lab_to_xyz(lab))


# ============================================================================
# CIEDE2000 Color Difference
# ============================================================================

def ciede2000_vectorized(lab1: np.ndarray, lab2: np.ndarray,
                         kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    """
    Vectorized CIEDE2000 calculation.
    
    Args:
        lab1: Array of shape (N, 3) - pixels in Lab
        lab2: Array of shape (M, 3) - palette in Lab
        
    Returns:
        distances: Array of shape (N, M)
    """
    lab1 = lab1[:, np.newaxis, :]
    lab2 = lab2[np.newaxis, :, :]
    
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    
    C_bar_7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar_7 / (C_bar_7 + 25.0**7)))
    
    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime_deg = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime_deg = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    h_diff = h2_prime_deg - h1_prime_deg
    delta_h_prime_deg = np.where(
        C1_prime * C2_prime == 0, 0,
        np.where(np.abs(h_diff) <= 180, h_diff,
                 np.where(h_diff > 180, h_diff - 360, h_diff + 360)))
    
    delta_H_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime_deg / 2.0))
    
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0
    
    h_sum = h1_prime_deg + h2_prime_deg
    h_abs_diff = np.abs(h1_prime_deg - h2_prime_deg)
    h_bar_prime_deg = np.where(
        C1_prime * C2_prime == 0, h_sum,
        np.where(h_abs_diff <= 180, h_sum / 2.0,
                 np.where(h_sum < 360, (h_sum + 360) / 2.0, (h_sum - 360) / 2.0)))
    
    T = (1.0 
         - 0.17 * np.cos(np.radians(h_bar_prime_deg - 30))
         + 0.24 * np.cos(np.radians(2 * h_bar_prime_deg))
         + 0.32 * np.cos(np.radians(3 * h_bar_prime_deg + 6))
         - 0.20 * np.cos(np.radians(4 * h_bar_prime_deg - 63)))
    
    delta_theta = 30.0 * np.exp(-((h_bar_prime_deg - 275) / 25.0) ** 2)
    
    C_bar_prime_7 = C_bar_prime ** 7
    R_C = 2.0 * np.sqrt(C_bar_prime_7 / (C_bar_prime_7 + 25.0**7))
    
    L_bar_prime_minus_50_sq = (L_bar_prime - 50) ** 2
    S_L = 1.0 + (0.015 * L_bar_prime_minus_50_sq) / np.sqrt(20 + L_bar_prime_minus_50_sq)
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T
    
    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C
    
    delta_E = np.sqrt(
        (delta_L_prime / (kL * S_L)) ** 2 +
        (delta_C_prime / (kC * S_C)) ** 2 +
        (delta_H_prime / (kH * S_H)) ** 2 +
        R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    )
    
    return delta_E


# ============================================================================
# Preprocessing Functions (for better palette utilization)
# ============================================================================

def auto_levels(image: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
    """
    Auto-levels: Stretch histogram to use full 0-255 range.
    Simple and effective for most dark/bright images.
    
    Args:
        image: Input image array
        clip_percent: Percentage to clip from each end (reduces outlier influence)
    """
    img_float = image.astype(np.float32)
    
    low = np.percentile(img_float, clip_percent)
    high = np.percentile(img_float, 100 - clip_percent)
    
    if high - low < 1:
        return image
    
    result = (img_float - low) * (255.0 / (high - low))
    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Histogram equalization in LAB color space.
    Equalizes only L channel to preserve color relationships.
    Good for images with poor contrast.
    """
    lab = rgb_to_lab(image)
    L = lab[..., 0]
    
    L_flat = L.flatten()
    hist, bins = np.histogram(L_flat, bins=256, range=(0, 100))
    cdf = hist.cumsum()
    cdf_normalized = cdf * 100.0 / cdf[-1]
    
    L_equalized = np.interp(L_flat, bins[:-1], cdf_normalized).reshape(L.shape)
    
    lab_eq = np.stack([L_equalized, lab[..., 1], lab[..., 2]], axis=-1)
    return lab_to_rgb(lab_eq)


def clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Local histogram equalization - preserves local details better than global.
    Best for images with varying lighting conditions.
    
    Args:
        clip_limit: Higher = more contrast (2.0-4.0 typical)
        grid_size: Number of tiles (8 = 8x8 grid)
    """
    lab = rgb_to_lab(image)
    L = lab[..., 0]
    h, w = L.shape
    
    tile_h = max(1, h // grid_size)
    tile_w = max(1, w // grid_size)
    
    # Process each tile
    result = np.zeros_like(L)
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * tile_h
            y2 = min((i + 1) * tile_h, h)
            x1 = j * tile_w
            x2 = min((j + 1) * tile_w, w)
            
            tile = L[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            
            # Build and clip histogram
            hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 100))
            clip_threshold = clip_limit * tile.size / 256
            excess = np.sum(np.maximum(hist - clip_threshold, 0))
            hist = np.minimum(hist, clip_threshold)
            hist = hist + excess / 256
            
            # Build CDF and apply
            cdf = hist.cumsum()
            if cdf[-1] > 0:
                cdf_normalized = cdf * 100.0 / cdf[-1]
                result[y1:y2, x1:x2] = np.interp(
                    tile.flatten(), bins[:-1], cdf_normalized
                ).reshape(tile.shape)
            else:
                result[y1:y2, x1:x2] = tile
    
    lab_result = np.stack([result, lab[..., 1], lab[..., 2]], axis=-1)
    return lab_to_rgb(lab_result)


def gamma_correction(image: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """
    Gamma correction.
    gamma < 1: Brighten dark areas (use 0.4-0.7 for dark images)
    gamma > 1: Darken bright areas (use 1.5-2.5 for bright images)
    """
    if gamma == 1.0:
        return image
    
    img_norm = image.astype(np.float32) / 255.0
    img_gamma = np.power(np.clip(img_norm, 0, 1), gamma)
    return (img_gamma * 255.0).astype(np.uint8)


def match_palette_lightness(image: np.ndarray, palette: list) -> np.ndarray:
    """
    Match image lightness distribution to palette's lightness distribution.
    This is the most theoretically correct approach for palette utilization.
    Ensures all palette colors have a chance to be used.
    """
    palette_arr = np.array(palette, dtype=np.float32)
    palette_lab = rgb_to_lab(palette_arr)
    palette_L = np.sort(palette_lab[:, 0])
    
    image_lab = rgb_to_lab(image)
    image_L = image_lab[..., 0]
    
    # Build image CDF
    image_L_flat = image_L.flatten()
    hist_img, bins_img = np.histogram(image_L_flat, bins=256, range=(0, 100))
    cdf_img = hist_img.cumsum().astype(np.float64)
    cdf_img /= cdf_img[-1]
    
    # Map through palette distribution
    palette_percentiles = np.linspace(0, 1, len(palette_L))
    image_percentiles = np.interp(image_L_flat, bins_img[:-1], cdf_img)
    new_L = np.interp(image_percentiles, palette_percentiles, palette_L)
    new_L = new_L.reshape(image_L.shape)
    
    lab_result = np.stack([new_L, image_lab[..., 1], image_lab[..., 2]], axis=-1)
    return lab_to_rgb(lab_result)


def auto_preprocess(image: np.ndarray, palette: list) -> np.ndarray:
    """
    Automatic preprocessing based on image analysis.
    Chooses the best method based on image characteristics.
    """
    lab = rgb_to_lab(image)
    L = lab[..., 0]
    
    mean_L = np.mean(L)
    std_L = np.std(L)
    
    print(f"  Image analysis: mean_L={mean_L:.1f}, std_L={std_L:.1f}")
    
    # Decision logic
    if std_L < 15:
        # Low contrast image
        print("  -> Low contrast detected, using CLAHE")
        return clahe(image, clip_limit=3.0)
    elif mean_L < 35:
        # Dark image
        print("  -> Dark image detected, using gamma + palette matching")
        result = gamma_correction(image, gamma=0.6)
        return match_palette_lightness(result, palette)
    elif mean_L > 70:
        # Bright image
        print("  -> Bright image detected, using gamma + palette matching")
        result = gamma_correction(image, gamma=1.5)
        return match_palette_lightness(result, palette)
    else:
        # Normal image - just match palette
        print("  -> Normal image, using palette matching")
        return match_palette_lightness(image, palette)


# ============================================================================
# Core Algorithm Implementation
# ============================================================================

def apply_palette_rgb(image: np.ndarray, palette: list) -> np.ndarray:
    """Apply palette using RGB Euclidean distance (fast)."""
    palette_arr = np.array(palette, dtype=np.float32)
    h, w, c = image.shape
    
    pixels = image.reshape(-1, 3).astype(np.float32)
    distances = np.sum(
        (pixels[:, np.newaxis, :] - palette_arr[np.newaxis, :, :]) ** 2,
        axis=2
    )
    
    nearest_indices = np.argmin(distances, axis=1)
    result = palette_arr[nearest_indices].astype(np.uint8)
    
    return result.reshape(h, w, c)


def apply_palette_ciede2000(image: np.ndarray, palette: list) -> np.ndarray:
    """Apply palette using CIEDE2000 (perceptually accurate, slower)."""
    palette_arr = np.array(palette, dtype=np.float32)
    h, w, c = image.shape
    
    print("  Converting to Lab...")
    pixels = image.reshape(-1, 3)
    pixels_lab = rgb_to_lab(pixels)
    palette_lab = rgb_to_lab(palette_arr)
    
    print("  Calculating CIEDE2000...")
    chunk_size = 10000
    n_pixels = pixels_lab.shape[0]
    nearest_indices = np.zeros(n_pixels, dtype=np.int32)
    
    for i in range(0, n_pixels, chunk_size):
        end = min(i + chunk_size, n_pixels)
        distances = ciede2000_vectorized(pixels_lab[i:end], palette_lab)
        nearest_indices[i:end] = np.argmin(distances, axis=1)
        print(f"\r  Progress: {(end / n_pixels) * 100:.1f}%", end="", flush=True)
    
    print()
    
    result = palette_arr[nearest_indices].astype(np.uint8)
    return result.reshape(h, w, c)


def apply_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    """Brightness adjustment (-100 to 200)."""
    if brightness == 0:
        return image
    factor = brightness / 100.0 + 1.0
    result = image.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast(image: np.ndarray, contrast: float) -> np.ndarray:
    """Contrast adjustment (-100 to 200)."""
    if contrast == 0:
        return image
    factor = contrast / 100.0 + 1.0
    offset = 128.0 * (1.0 - factor)
    result = image.astype(np.float32) * factor + offset
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_saturation(image: np.ndarray, saturation: float) -> np.ndarray:
    """Saturation adjustment (-100 to 200)."""
    if saturation == 0:
        return image
    factor = saturation / 100.0 + 1.0
    img_float = image.astype(np.float32)
    gray = (0.3086 * img_float[:, :, 0] + 
            0.6094 * img_float[:, :, 1] + 
            0.0820 * img_float[:, :, 2])[:, :, np.newaxis]
    result = factor * img_float + (1.0 - factor) * gray
    return np.clip(result, 0, 255).astype(np.uint8)


def downscale_image(image: Image.Image, pixel_size: int) -> Image.Image:
    """Downscale by pixel_size factor."""
    if pixel_size <= 1:
        return image
    new_w = max(1, image.width // pixel_size)
    new_h = max(1, image.height // pixel_size)
    return image.resize((new_w, new_h), Image.Resampling.NEAREST)


def upscale_image(image: Image.Image, target_size: tuple) -> Image.Image:
    """Upscale back to original size."""
    return image.resize(target_size, Image.Resampling.NEAREST)


# ============================================================================
# Main Conversion Pipeline
# ============================================================================

def convert_to_pixel_art(
    input_path: str,
    output_path: str,
    pixel_size: int = 8,
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    palette_name: str = "pico8",
    use_palette: bool = True,
    scale_up: bool = False,
    use_ciede2000: bool = False,
    preprocess: str = None,
    gamma: float = None,
) -> None:
    """Convert an image to pixel art."""
    
    print(f"Loading: {input_path}")
    img = Image.open(input_path)
    original_size = img.size
    
    # Convert to RGB
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    print(f"Original: {original_size[0]}x{original_size[1]}")
    
    # Downscale
    img = downscale_image(img, pixel_size)
    print(f"Downscaled: {img.width}x{img.height}")
    
    img_array = np.array(img)
    
    # Get palette for preprocessing
    if palette_name not in PALETTES:
        print(f"Warning: Unknown palette '{palette_name}', using 'pico8'")
        palette_name = "pico8"
    palette = PALETTES[palette_name]
    
    # Apply preprocessing
    if preprocess:
        print(f"Preprocessing: {preprocess}")
        if preprocess == "auto":
            img_array = auto_preprocess(img_array, palette)
        elif preprocess == "histogram":
            img_array = histogram_equalization(img_array)
        elif preprocess == "clahe":
            img_array = clahe(img_array)
        elif preprocess == "match_palette":
            img_array = match_palette_lightness(img_array, palette)
        elif preprocess == "auto_levels":
            img_array = auto_levels(img_array)
    
    # Apply gamma if specified
    if gamma is not None:
        print(f"Gamma: {gamma}")
        img_array = gamma_correction(img_array, gamma)
    
    # Apply filters
    if brightness != 0:
        print(f"Brightness: {brightness}")
        img_array = apply_brightness(img_array, brightness)
    if contrast != 0:
        print(f"Contrast: {contrast}")
        img_array = apply_contrast(img_array, contrast)
    if saturation != 0:
        print(f"Saturation: {saturation}")
        img_array = apply_saturation(img_array, saturation)
    
    # Apply palette
    if use_palette:
        method = "CIEDE2000" if use_ciede2000 else "RGB"
        print(f"Palette ({method}): {palette_name} ({len(palette)} colors)")
        if use_ciede2000:
            img_array = apply_palette_ciede2000(img_array, palette)
        else:
            img_array = apply_palette_rgb(img_array, palette)
    
    result = Image.fromarray(img_array)
    
    if scale_up:
        result = upscale_image(result, original_size)
        print(f"Scaled up: {result.width}x{result.height}")
    
    result.save(output_path)
    print(f"Saved: {output_path}")


def list_palettes():
    """Print available palettes."""
    print("\nAvailable palettes:")
    print("-" * 40)
    for name, colors in PALETTES.items():
        print(f"  {name:15} ({len(colors):2} colors)")
    print()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert images to pixel art",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", nargs="?", help="Input image")
    parser.add_argument("-o", "--output", default="output.png")
    parser.add_argument("-p", "--pixel-size", type=int, default=8)
    parser.add_argument("-b", "--brightness", type=float, default=0)
    parser.add_argument("-c", "--contrast", type=float, default=0)
    parser.add_argument("-s", "--saturation", type=float, default=0)
    parser.add_argument("--palette", default="pico8")
    parser.add_argument("--no-palette", action="store_true")
    parser.add_argument("--scale-up", action="store_true")
    parser.add_argument("--ciede2000", action="store_true",
                        help="Use CIEDE2000 color distance")
    parser.add_argument("--preprocess", 
                        choices=["auto", "histogram", "clahe", "match_palette", "auto_levels"],
                        help="Preprocessing method for better palette utilization")
    parser.add_argument("--gamma", type=float,
                        help="Gamma correction (0.4-0.7 for dark, 1.5-2.5 for bright)")
    parser.add_argument("--list-palettes", action="store_true")
    
    args = parser.parse_args()
    
    if args.list_palettes:
        list_palettes()
        return
    
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)
    
    try:
        convert_to_pixel_art(
            input_path=args.input,
            output_path=args.output,
            pixel_size=args.pixel_size,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            palette_name=args.palette,
            use_palette=not args.no_palette,
            scale_up=args.scale_up,
            use_ciede2000=args.ciede2000,
            preprocess=args.preprocess,
            gamma=args.gamma,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
