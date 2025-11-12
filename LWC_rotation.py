"""
LWC Rotation Normalization

Mirrors the MSD rotation workflow: preprocess to edges, scan angles, match
against reference masks using normalized cross-correlation, show plots, and
apply the selected rotation.

Author: John Anthony Kadian (adapted for LWC)
Date: 10/12/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def preprocess_for_lwc_comparison(image, face_type: str):
    """
    Mimic LWC ML feature extraction (CoinImage.applymask):
    - Grayscale -> Gaussian blur (sigma=1.5)
    - Sobel (ksize=5) in X and Y
    - Gradient magnitude (float) -> normalize to 0..255
    - Light median filter to reduce salt-and-pepper
    - Mask to coin circle
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    smoothed = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.5, sigmaY=1.5)
    dx = cv2.Sobel(smoothed, cv2.CV_16S, 1, 0, ksize=5)
    dy = cv2.Sobel(smoothed, cv2.CV_16S, 0, 1, ksize=5)
    dx32 = dx.astype(np.float32)
    dy32 = dy.astype(np.float32)
    mag = np.sqrt(dx32 * dx32 + dy32 * dy32)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_u8 = cv2.medianBlur(mag_u8, 3)

    h, w = image.shape[:2]
    circ = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circ, (w//2, h//2), 487, 255, -1)
    edges = cv2.bitwise_and(mag_u8, circ)
    return edges


def load_lwc_reference_mask(face_type):
    """
    Load LWC reference mask for the given face type.
    - Obverse: lwc_obverse_mask.jpg (combined head + coin circle)
    - Reverse: lwc_reverse_mask.jpg (combined wheat stalks + coin circle)
    """
    mask_dir = os.path.join('LincolnCent', 'CustomMasks')
    if face_type.lower() == 'obverse':
        mask_path = os.path.join(mask_dir, 'lwc_obverse_mask.jpg')
    elif face_type.lower() == 'reverse':
        mask_path = os.path.join(mask_dir, 'lwc_reverse_mask.jpg')
    else:
        raise ValueError("face_type must be 'obverse' or 'reverse'")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load LWC mask for {face_type}")
        return None
    # Invert mask per updated requirement
    mask = cv2.bitwise_not(mask)
    print(f"Loaded LWC {face_type} reference mask (inverted): {mask.shape}")
    return mask


def find_optimal_rotation(coin_image, reference_mask, face_type, angle_range=180, step=1.0):
    """
    Find optimal rotation using template matching with normalized cross-correlation.
    Uses optimized two-stage search: 5° coarse search, then 0.5° fine search around best angle.
    Note: Intentionally mirrors MSD printouts/logic.
    """
    print(f"Finding optimal rotation for {face_type}")

    coin_edges = preprocess_for_lwc_comparison(coin_image, face_type)
    print(f"Preprocessed coin using Sobel gradient magnitude pipeline")
    print(f"Using template matching with normalized cross-correlation (MSD team)")

    best_angle = 0
    best_score = -1
    scores = []  # will hold coarse + fine
    angles = []

    # Stage 1: Coarse search with 5° steps
    coarse_step = 5.0
    coarse_angles = np.arange(-angle_range/2, angle_range/2 + coarse_step, coarse_step)
    
    best_coarse_angle = 0
    best_coarse_score = -1
    
    coarse_scores = []
    coarse_angles_list = []
    for angle in coarse_angles:
        if abs(angle) < 0.1:
            rotated_edges = coin_edges
        else:
            h, w = coin_edges.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_edges = cv2.warpAffine(coin_edges, rotation_matrix, (w, h), borderValue=0)

        result = cv2.matchTemplate(rotated_edges, reference_mask, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        correlation_score = max_val

        if correlation_score > best_coarse_score or best_coarse_score == -1:
            best_coarse_score = correlation_score
            best_coarse_angle = angle
        coarse_scores.append(correlation_score)
        coarse_angles_list.append(angle)
    
    # Stage 2: Fine search with 0.5° steps around best coarse angle
    fine_start = best_coarse_angle - coarse_step
    fine_end = best_coarse_angle + coarse_step
    fine_angles = np.arange(fine_start, fine_end + 0.5, 0.5)
    
    for angle in fine_angles:
        if abs(angle) < 0.1:
            rotated_edges = coin_edges
        else:
            h, w = coin_edges.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_edges = cv2.warpAffine(coin_edges, rotation_matrix, (w, h), borderValue=0)

        result = cv2.matchTemplate(rotated_edges, reference_mask, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        correlation_score = max_val

        scores.append(correlation_score)
        angles.append(angle)

        if correlation_score > best_score or best_score == -1:
            best_score = correlation_score
            best_angle = angle

    print(f"Best rotation: {best_angle:.1f}° (max correlation: {best_score:.4f})")

    # Prepend coarse to provide full horizontal range in downstream plots
    if len(coarse_angles_list) > 0:
        angles = list(coarse_angles_list) + angles
        scores = list(coarse_scores) + scores

    # create_rotation_visualization(coin_image, coin_edges, reference_mask, best_angle,
    #                               best_score, face_type, angles, scores)
    return best_angle, best_score, scores, angles


def create_rotation_visualization(coin_image, coin_edges, reference_mask, best_angle,
                                  best_score, face_type, angles, scores):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    coin_rgb = cv2.cvtColor(coin_image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(coin_rgb)
    axes[0, 0].set_title(f"Original {face_type.title()}\n(1000x1000, 97.5% diameter)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(coin_edges, cmap='gray')
    axes[0, 1].set_title("LWC Preprocessed Edges\n(Sobel Gradient Magnitude)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(reference_mask, cmap='gray')
    axes[0, 2].set_title("Reference Mask\n(Regions of Interest)")
    axes[0, 2].axis('off')

    axes[0, 3].plot(angles, scores, 'b-', linewidth=2)
    axes[0, 3].axvline(x=best_angle, color='r', linestyle='--', linewidth=2, label=f'Best: {best_angle:.1f}°')
    axes[0, 3].set_xlabel('Rotation Angle (degrees)')
    axes[0, 3].set_ylabel('Correlation Score')
    axes[0, 3].set_title('Rotation Score vs Angle')
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].legend()

    if abs(best_angle) > 0.1:
        h, w = coin_edges.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated_edges = cv2.warpAffine(coin_edges, rotation_matrix, (w, h), borderValue=0)
    else:
        rotated_edges = coin_edges
    axes[1, 0].imshow(rotated_edges, cmap='gray')
    axes[1, 0].set_title(f"Rotated Edges\n({best_angle:.1f}°)")
    axes[1, 0].axis('off')

    inverted_mask = cv2.bitwise_not(reference_mask)
    masked_edges = cv2.bitwise_and(rotated_edges, inverted_mask)
    axes[1, 1].imshow(masked_edges, cmap='gray')
    axes[1, 1].set_title("Mask Applied to Rotated Edges")
    axes[1, 1].axis('off')

    if abs(best_angle) > 0.1:
        h, w = coin_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated_coin = cv2.warpAffine(coin_image, rotation_matrix, (w, h), borderValue=(255, 255, 255))
    else:
        rotated_coin = coin_image
    rotated_coin_rgb = cv2.cvtColor(rotated_coin, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(rotated_coin_rgb)
    axes[1, 2].set_title(f"Final Rotated {face_type.title()}\n({best_angle:.1f}°)")
    axes[1, 2].axis('off')

    comparison = np.hstack([coin_rgb, rotated_coin_rgb])
    axes[1, 3].imshow(comparison)
    axes[1, 3].set_title("Comparison\n(Original | Rotated)")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.show()


def apply_rotation_normalization(coin_image, angle):
    if abs(angle) < 0.5:
        return coin_image
    h, w = coin_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_angle) + (w * cos_angle))
    new_h = int((h * cos_angle) + (w * sin_angle))
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(coin_image, rotation_matrix, (new_w, new_h), borderValue=(255, 255, 255))
    start_x = (new_w - 1000) // 2
    start_y = (new_h - 1000) // 2
    return rotated[start_y:start_y+1000, start_x:start_x+1000]


def rotate_coin(image, face_type):
    print(f"Rotating {face_type}")
    reference_mask = load_lwc_reference_mask(face_type)
    if reference_mask is None:
        print("No reference mask available, returning original image")
        return image
    optimal_angle = find_optimal_rotation(image, reference_mask, face_type)
    if abs(optimal_angle) > 0.5:
        rotated_image = apply_rotation_normalization(image, optimal_angle)
        print(f"Applied rotation: {optimal_angle:.1f}°")
        return rotated_image
    else:
        print("No rotation needed")
        return image


def rotate_coin_pair(obverse_image, reverse_image):
    print("LWC rotation normalization")
    rotated_obverse = rotate_coin(obverse_image, "obverse")
    rotated_reverse = rotate_coin(reverse_image, "reverse")
    print("\nRotation complete")
    return rotated_obverse, rotated_reverse


