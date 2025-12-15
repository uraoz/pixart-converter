#!/usr/bin/env python3
"""
Pixel Art Converter
Usage:
    python pixel_art_converter.py input.png [options]

Options:
    -o, --output        Output filename (default: output.png)
    -p, --pixel-size    Pixel size for downscaling (default: 1)
    -b, --brightness    Brightness adjustment -100 to 200 (default: 0)
    -c, --contrast      Contrast adjustment -100 to 200 (default: 0)
    -s, --saturation    Saturation adjustment -100 to 200 (default: 0)
    --palette           Palette name: pico8, lost_century, sunset8, twilight5, hollow, default (default: default)
    --no-palette        Disable palette mapping
    --scale-up          Scale output back to original size
    --ciede2000         Use CIEDE2000 color distance (slower but perceptually accurate)
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

PALETTES = {
    "pico8": [
        [0, 0, 0],
        [29, 43, 83],
        [126, 37, 83],
        [0, 135, 81],
        [171, 82, 54],
        [95, 87, 79],
        [194, 195, 199],
        [255, 241, 232],
        [255, 0, 77],
        [255, 163, 0],
        [255, 236, 39],
        [0, 228, 54],
        [41, 173, 255],
        [131, 118, 156],
        [255, 119, 168],
        [255, 204, 170],
    ],
    "lost_century": [
        [209, 177, 135],
        [199, 123, 88],
        [174, 93, 64],
        [121, 68, 74],
        [75, 61, 68],
        [186, 145, 88],
        [146, 116, 65],
        [77, 69, 57],
        [119, 116, 59],
        [179, 165, 85],
        [210, 201, 165],
        [140, 171, 161],
        [75, 114, 110],
        [87, 72, 82],
        [132, 120, 117],
        [171, 155, 142],
    ],
    "sunset8": [
        [255, 255, 120],
        [255, 214, 71],
        [255, 194, 71],
        [255, 169, 54],
        [255, 139, 111],
        [230, 117, 149],
        [154, 99, 144],
        [70, 70, 120],
    ],
    "twilight5": [
        [251, 187, 173],
        [238, 134, 149],
        [74, 122, 150],
        [51, 63, 88],
        [41, 40, 49],
    ],
    "hollow": [
        [15, 15, 27],
        [86, 90, 117],
        [198, 183, 190],
        [250, 251, 246],
    ],
    "gameboy": [
        [15, 56, 15],
        [48, 98, 48],
        [139, 172, 15],
        [155, 188, 15],
    ],
    "cga": [
        [0, 0, 0],
        [0, 170, 170],
        [170, 0, 170],
        [170, 170, 170],
    ],
    # Additional palettes
    "endesga32": [
        [190, 74, 47],
        [215, 118, 67],
        [234, 212, 170],
        [228, 166, 114],
        [184, 111, 80],
        [115, 62, 57],
        [62, 39, 49],
        [162, 38, 51],
        [228, 59, 68],
        [247, 118, 34],
        [254, 174, 52],
        [254, 231, 97],
        [99, 199, 77],
        [62, 137, 72],
        [38, 92, 66],
        [25, 60, 62],
        [18, 78, 137],
        [0, 153, 219],
        [44, 232, 245],
        [192, 203, 220],
        [139, 155, 180],
        [90, 105, 136],
        [58, 68, 102],
        [38, 43, 68],
        [24, 20, 37],
        [255, 0, 68],
        [104, 56, 108],
        [181, 80, 136],
        [246, 117, 122],
        [232, 183, 150],
        [194, 133, 105],
        [140, 94, 88],
    ],
    "default": [
        [11, 11, 18],
        [26, 26, 46], 
        [46, 46, 82],
        [74, 74, 120],
        [120, 120, 160],
        [184, 176, 160],
        [212, 168, 48],
        [232, 224, 208],
    ],
}


# ============================================================================
# Color Space Conversion Functions
# ============================================================================

def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to XYZ color space.
    Assumes sRGB with D65 illuminant.
    
    Args:
        rgb: Array of shape (..., 3) with values 0-255
    
    Returns:
        xyz: Array of shape (..., 3)
    """
    # Normalize to 0-1
    rgb_norm = rgb.astype(np.float64) / 255.0
    
    # Apply sRGB gamma correction (inverse companding)
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(
        mask,
        ((rgb_norm + 0.055) / 1.055) ** 2.4,
        rgb_norm / 12.92
    )
    
    # RGB to XYZ transformation matrix (sRGB, D65)
    # http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = np.einsum('...j,ij->...i', rgb_linear, M) * 100.0
    return xyz


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to CIELAB color space.
    Uses D65 illuminant reference white.
    
    Args:
        xyz: Array of shape (..., 3)
    
    Returns:
        lab: Array of shape (..., 3) with L*, a*, b*
    """
    # D65 reference white
    ref_white = np.array([95.047, 100.000, 108.883])
    
    xyz_norm = xyz / ref_white
    
    # Apply f function
    epsilon = 216.0 / 24389.0  # 0.008856
    kappa = 24389.0 / 27.0     # 903.3
    
    mask = xyz_norm > epsilon
    f = np.where(
        mask,
        np.cbrt(xyz_norm),
        (kappa * xyz_norm + 16.0) / 116.0
    )
    
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to CIELAB."""
    return xyz_to_lab(rgb_to_xyz(rgb))


# ============================================================================
# CIEDE2000 Color Difference
# ============================================================================

def ciede2000_single(lab1: np.ndarray, lab2: np.ndarray, 
                     kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
    """
    Calculate CIEDE2000 color difference between two Lab colors.
    
    Reference: 
    - Sharma, G., Wu, W., & Dalal, E. N. (2005). 
      "The CIEDE2000 color-difference formula: Implementation notes, 
       supplementary test data, and mathematical observations."
    
    Args:
        lab1, lab2: Lab color values [L, a, b]
        kL, kC, kH: Weighting factors (default 1.0 for most applications)
    
    Returns:
        deltaE: Color difference value
    """
    L1, a1, b1 = lab1[0], lab1[1], lab1[2]
    L2, a2, b2 = lab2[0], lab2[1], lab2[2]
    
    # Step 1: Calculate C'i and h'i
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    
    C_bar_7 = C_bar ** 7
    G = 0.5 * (1.0 - math.sqrt(C_bar_7 / (C_bar_7 + 25.0**7)))
    
    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)
    
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = math.atan2(b1, a1_prime) % (2 * math.pi)
    h2_prime = math.atan2(b2, a2_prime) % (2 * math.pi)
    
    # Convert to degrees
    h1_prime_deg = math.degrees(h1_prime)
    h2_prime_deg = math.degrees(h2_prime)
    
    # Step 2: Calculate deltaL', deltaC', deltaH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    h_diff = h2_prime_deg - h1_prime_deg
    
    if C1_prime * C2_prime == 0:
        delta_h_prime_deg = 0
    elif abs(h_diff) <= 180:
        delta_h_prime_deg = h_diff
    elif h_diff > 180:
        delta_h_prime_deg = h_diff - 360
    else:
        delta_h_prime_deg = h_diff + 360
    
    delta_H_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime_deg / 2.0))
    
    # Step 3: Calculate CIEDE2000
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0
    
    if C1_prime * C2_prime == 0:
        h_bar_prime_deg = h1_prime_deg + h2_prime_deg
    elif abs(h1_prime_deg - h2_prime_deg) <= 180:
        h_bar_prime_deg = (h1_prime_deg + h2_prime_deg) / 2.0
    elif h1_prime_deg + h2_prime_deg < 360:
        h_bar_prime_deg = (h1_prime_deg + h2_prime_deg + 360) / 2.0
    else:
        h_bar_prime_deg = (h1_prime_deg + h2_prime_deg - 360) / 2.0
    
    T = (1.0 
         - 0.17 * math.cos(math.radians(h_bar_prime_deg - 30))
         + 0.24 * math.cos(math.radians(2 * h_bar_prime_deg))
         + 0.32 * math.cos(math.radians(3 * h_bar_prime_deg + 6))
         - 0.20 * math.cos(math.radians(4 * h_bar_prime_deg - 63)))
    
    delta_theta = 30.0 * math.exp(-((h_bar_prime_deg - 275) / 25.0) ** 2)
    
    C_bar_prime_7 = C_bar_prime ** 7
    R_C = 2.0 * math.sqrt(C_bar_prime_7 / (C_bar_prime_7 + 25.0**7))
    
    L_bar_prime_minus_50_sq = (L_bar_prime - 50) ** 2
    S_L = 1.0 + (0.015 * L_bar_prime_minus_50_sq) / math.sqrt(20 + L_bar_prime_minus_50_sq)
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T
    
    R_T = -math.sin(math.radians(2 * delta_theta)) * R_C
    
    delta_E = math.sqrt(
        (delta_L_prime / (kL * S_L)) ** 2 +
        (delta_C_prime / (kC * S_C)) ** 2 +
        (delta_H_prime / (kH * S_H)) ** 2 +
        R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    )
    
    return delta_E


def ciede2000_vectorized(lab1: np.ndarray, lab2: np.ndarray,
                         kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    """
    Vectorized CIEDE2000 calculation for multiple color pairs.
    
    Args:
        lab1: Array of shape (N, 3) - first set of Lab colors
        lab2: Array of shape (M, 3) - second set of Lab colors (palette)
        
    Returns:
        distances: Array of shape (N, M) with CIEDE2000 distances
    """
    # Expand dimensions for broadcasting: (N, 1, 3) vs (1, M, 3)
    lab1 = lab1[:, np.newaxis, :]  # (N, 1, 3)
    lab2 = lab2[np.newaxis, :, :]  # (1, M, 3)
    
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    
    # Step 1
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    
    C_bar_7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar_7 / (C_bar_7 + 25.0**7)))
    
    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.arctan2(b1, a1_prime)
    h2_prime = np.arctan2(b2, a2_prime)
    
    # Wrap to 0-360
    h1_prime_deg = np.degrees(h1_prime) % 360
    h2_prime_deg = np.degrees(h2_prime) % 360
    
    # Step 2
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    h_diff = h2_prime_deg - h1_prime_deg
    
    # Handle delta_h_prime
    delta_h_prime_deg = np.where(
        C1_prime * C2_prime == 0,
        0,
        np.where(
            np.abs(h_diff) <= 180,
            h_diff,
            np.where(h_diff > 180, h_diff - 360, h_diff + 360)
        )
    )
    
    delta_H_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime_deg / 2.0))
    
    # Step 3
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0
    
    # Handle h_bar_prime
    h_sum = h1_prime_deg + h2_prime_deg
    h_abs_diff = np.abs(h1_prime_deg - h2_prime_deg)
    
    h_bar_prime_deg = np.where(
        C1_prime * C2_prime == 0,
        h_sum,
        np.where(
            h_abs_diff <= 180,
            h_sum / 2.0,
            np.where(h_sum < 360, (h_sum + 360) / 2.0, (h_sum - 360) / 2.0)
        )
    )
    
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
    
    term1 = (delta_L_prime / (kL * S_L)) ** 2
    term2 = (delta_C_prime / (kC * S_C)) ** 2
    term3 = (delta_H_prime / (kH * S_H)) ** 2
    term4 = R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    
    delta_E = np.sqrt(term1 + term2 + term3 + term4)
    
    return delta_E


# ============================================================================
# Core Algorithm Implementation
# ============================================================================

def color_distance_rgb(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Calculate squared Euclidean distance in RGB space.
    Matches the original JS implementation (no square root for performance).
    """
    diff = c1.astype(np.float32) - c2.astype(np.float32)
    return np.sum(diff ** 2)


def apply_palette_rgb(image: np.ndarray, palette: list) -> np.ndarray:
    """
    Apply palette using RGB Euclidean distance (original algorithm).
    """
    palette_arr = np.array(palette, dtype=np.float32)
    h, w, c = image.shape
    
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # Calculate distances to all palette colors
    distances = np.sum(
        (pixels[:, np.newaxis, :] - palette_arr[np.newaxis, :, :]) ** 2,
        axis=2
    )
    
    nearest_indices = np.argmin(distances, axis=1)
    result = palette_arr[nearest_indices].astype(np.uint8)
    
    return result.reshape(h, w, c)


def apply_palette_ciede2000(image: np.ndarray, palette: list) -> np.ndarray:
    """
    Apply palette using CIEDE2000 color distance (perceptually accurate).
    """
    palette_arr = np.array(palette, dtype=np.float32)
    h, w, c = image.shape
    
    print("  Converting to Lab color space...")
    pixels = image.reshape(-1, 3)
    pixels_lab = rgb_to_lab(pixels)
    palette_lab = rgb_to_lab(palette_arr)
    
    print("  Calculating CIEDE2000 distances...")
    # Process in chunks to manage memory
    chunk_size = 10000
    n_pixels = pixels_lab.shape[0]
    nearest_indices = np.zeros(n_pixels, dtype=np.int32)
    
    for i in range(0, n_pixels, chunk_size):
        end = min(i + chunk_size, n_pixels)
        chunk = pixels_lab[i:end]
        distances = ciede2000_vectorized(chunk, palette_lab)
        nearest_indices[i:end] = np.argmin(distances, axis=1)
        
        # Progress indicator
        progress = (end / n_pixels) * 100
        print(f"\r  Progress: {progress:.1f}%", end="", flush=True)
    
    print()  # Newline after progress
    
    result = palette_arr[nearest_indices].astype(np.uint8)
    return result.reshape(h, w, c)


def apply_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    """Apply brightness adjustment. brightness: -100 to 200"""
    if brightness == 0:
        return image
    
    factor = brightness / 100.0 + 1.0
    result = image.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast(image: np.ndarray, contrast: float) -> np.ndarray:
    """Apply contrast adjustment. contrast: -100 to 200"""
    if contrast == 0:
        return image
    
    factor = contrast / 100.0 + 1.0
    offset = 128.0 * (1.0 - factor)
    result = image.astype(np.float32) * factor + offset
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_saturation(image: np.ndarray, saturation: float) -> np.ndarray:
    """Apply saturation adjustment. saturation: -100 to 200"""
    if saturation == 0:
        return image
    
    factor = saturation / 100.0 + 1.0
    img_float = image.astype(np.float32)
    
    # Calculate grayscale using rec601 luma coefficients
    gray = (0.3086 * img_float[:, :, 0] + 
            0.6094 * img_float[:, :, 1] + 
            0.0820 * img_float[:, :, 2])
    gray = gray[:, :, np.newaxis]
    
    result = factor * img_float + (1.0 - factor) * gray
    return np.clip(result, 0, 255).astype(np.uint8)


def downscale_image(image: Image.Image, pixel_size: int) -> Image.Image:
    """Downscale image by pixel_size factor using nearest neighbor."""
    if pixel_size <= 1:
        return image
    
    new_width = max(1, image.width // pixel_size)
    new_height = max(1, image.height // pixel_size)
    
    return image.resize((new_width, new_height), Image.Resampling.NEAREST)


def upscale_image(image: Image.Image, target_size: tuple) -> Image.Image:
    """Upscale image back to target size using nearest neighbor."""
    return image.resize(target_size, Image.Resampling.NEAREST)


# ============================================================================
# Main Conversion Pipeline
# ============================================================================

def convert_to_pixel_art(
    input_path: str,
    output_path: str,
    pixel_size: int = 1,
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    palette_name: str = "default",
    use_palette: bool = True,
    scale_up: bool = False,
    use_ciede2000: bool = False,
) -> None:
    """
    Convert an image to pixel art.
    
    Args:
        input_path: Path to input image
        output_path: Path to output image
        pixel_size: Downscale factor (1 = no downscale)
        brightness: Brightness adjustment (-100 to 200)
        contrast: Contrast adjustment (-100 to 200)
        saturation: Saturation adjustment (-100 to 200)
        palette_name: Name of palette to use
        use_palette: Whether to apply palette mapping
        scale_up: Whether to scale output back to original size
        use_ciede2000: Use CIEDE2000 instead of RGB Euclidean distance
    """
    # Load image
    print(f"Loading: {input_path}")
    img = Image.open(input_path)
    original_size = img.size
    
    # Convert to RGB if necessary
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    print(f"Original size: {original_size[0]}x{original_size[1]}")
    
    # Downscale
    img = downscale_image(img, pixel_size)
    print(f"Downscaled to: {img.width}x{img.height}")
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Apply filters
    if brightness != 0:
        print(f"Applying brightness: {brightness}")
        img_array = apply_brightness(img_array, brightness)
    
    if contrast != 0:
        print(f"Applying contrast: {contrast}")
        img_array = apply_contrast(img_array, contrast)
    
    if saturation != 0:
        print(f"Applying saturation: {saturation}")
        img_array = apply_saturation(img_array, saturation)
    
    # Apply palette
    if use_palette:
        if palette_name not in PALETTES:
            print(f"Warning: Unknown palette '{palette_name}', using 'default'")
            palette_name = "default"
        
        palette = PALETTES[palette_name]
        
        if use_ciede2000:
            print(f"Applying palette (CIEDE2000): {palette_name} ({len(palette)} colors)")
            img_array = apply_palette_ciede2000(img_array, palette)
        else:
            print(f"Applying palette (RGB): {palette_name} ({len(palette)} colors)")
            img_array = apply_palette_rgb(img_array, palette)
    
    # Convert back to PIL Image
    result = Image.fromarray(img_array)
    
    # Optionally scale back up
    if scale_up:
        result = upscale_image(result, original_size)
        print(f"Scaled up to: {result.width}x{result.height}")
    
    # Save
    result.save(output_path)
    print(f"Saved: {output_path}")
    print(f"Final size: {result.width}x{result.height}")


def list_palettes():
    """Print available palettes with color counts."""
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
        description="Convert images to pixel art with palette mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("input", nargs="?", help="Input image file")
    parser.add_argument("-o", "--output", default="output.png",
                        help="Output filename (default: output.png)")
    parser.add_argument("-p", "--pixel-size", type=int, default=1,
                        help="Pixel size for downscaling (default: 1)")
    parser.add_argument("-b", "--brightness", type=float, default=0,
                        help="Brightness: -100 to 200 (default: 0)")
    parser.add_argument("-c", "--contrast", type=float, default=0,
                        help="Contrast: -100 to 200 (default: 0)")
    parser.add_argument("-s", "--saturation", type=float, default=0,
                        help="Saturation: -100 to 200 (default: 0)")
    parser.add_argument("--palette", default="default",
                        help="Palette name (default: default)")
    parser.add_argument("--no-palette", action="store_true",
                        help="Disable palette mapping")
    parser.add_argument("--scale-up", action="store_true",
                        help="Scale output back to original size")
    parser.add_argument("--ciede2000", action="store_true",
                        help="Use CIEDE2000 color distance (slower but perceptually accurate)")
    parser.add_argument("--list-palettes", action="store_true",
                        help="List available palettes and exit")
    
    args = parser.parse_args()
    
    if args.list_palettes:
        list_palettes()
        return
    
    if not args.input:
        parser.print_help()
        print("\nError: Input file required")
        sys.exit(1)
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    
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
    )
    

if __name__ == "__main__":
    main()
