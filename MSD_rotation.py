"""
MSD Rotation Normalization

Handles rotation normalization for Morgan Silver Dollar coins by comparing
processed coin images against reference masks using the same preprocessing
pipeline as the MSD grading model.

Author: John Anthony Kadian
Date: 9/15/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import morphology


def preprocess_for_msd_comparison(image):
    """
    Preprocess to Sobel gradient magnitude (mirrors LWC model-aligned pipeline):
    - Gray -> Gaussian blur (sigma=1.5)
    - Sobel X/Y (ksize=5)
    - Gradient magnitude normalized to 0..255
    - Light median filter
    - Optional circle mask to suppress outer artifacts
    Returns single-channel edges suitable for template matching.
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

    # constrain to coin circle (radius 487)
    h, w = image.shape[:2]
    circ = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circ, (w//2, h//2), 487, 255, -1)
    edges = cv2.bitwise_and(mag_u8, circ)
    return edges


def load_msd_reference_mask(face_type):
    """
    Load the appropriate MSD reference mask for the given face type
    
    Args:
        face_type (str): Either "obverse" or "reverse"
    
    Returns:
        numpy.ndarray: The reference mask image (1000x1000)
    """
    # masks used from MorganSilverDollar/Morgan-Dollar-main/CustomMasks/ (MSD project team)
    mask_dir = "MorganSilverDollar/Morgan-Dollar-main/CustomMasks"
    
    if face_type.lower() == "obverse":
        mask_path = os.path.join(mask_dir, "obv_flat.jpg")
    elif face_type.lower() == "reverse":
        mask_path = os.path.join(mask_dir, "rev_flat.jpg")
    else:
        raise ValueError("face_type must be 'obverse' or 'reverse'")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load mask {mask_path}")
        return None
    # Invert mask per updated requirement
    mask = cv2.bitwise_not(mask)
    print(f"Loaded {face_type} reference mask (inverted): {mask.shape}")
    return mask


def find_optimal_rotation(coin_image, reference_mask, face_type, angle_range=180, step=1.0):
    """
    Find the optimal rotation angle using template matching with normalized cross-correlation.
    Uses optimized two-stage search: 5° coarse search, then 0.5° fine search around best angle.
    
    Args:
        coin_image: The processed coin image (1000x1000) with white background
        reference_mask: The reference mask (1000x1000)
        face_type: "obverse" or "reverse" for display purposes
        angle_range: Range of angles to test (default 180 degrees)
        step: Ignored - uses optimized 5°→0.5° search internally
    
    Returns:
        tuple: (best_angle, best_score) - optimal rotation angle and correlation score
    """
    print(f"Finding optimal rotation for {face_type}")
    
    # preprocess coin image using MSD pipeline
    coin_edges = preprocess_for_msd_comparison(coin_image)
    print(f"Preprocessed coin using Sobel magnitude (model-aligned)")
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

    # display visualized results - DISABLED
    # create_rotation_visualization(coin_image, coin_edges, reference_mask, best_angle, 
    #                              best_score, face_type, angles, scores)
    pass
    
    return best_angle, best_score, scores, angles


# Visualization function disabled - no longer needed
# def create_rotation_visualization(...):
#     pass


def apply_rotation_normalization(coin_image, angle):
    """
    Apply rotation normalization to the coin image given the angle.
    
    Args:
        coin_image: The 1000x1000 processed coin image
        angle: Rotation angle in degrees
    
    Returns:
        numpy.ndarray: Rotated image with white background preserved
    """
    if abs(angle) < 0.5:  # Skip tiny rotations
        return coin_image
    
    h, w = coin_image.shape[:2]
    center = (w // 2, h // 2)
    
    # get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # calculate and adjust new dimensions
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_angle) + (w * cos_angle))
    new_h = int((h * cos_angle) + (w * sin_angle))
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    # apply rotation
    rotated = cv2.warpAffine(coin_image, rotation_matrix, (new_w, new_h), 
                            borderValue=(255, 255, 255))
    
    # crop back to 1000x1000 square with coin centered
    start_x = (new_w - 1000) // 2
    start_y = (new_h - 1000) // 2
    final_image = rotated[start_y:start_y+1000, start_x:start_x+1000]
    
    return final_image


def rotate_coin(image, face_type):
    """
    Rotate a coin to optimal orientation using MSD reference mask.
    
    Args:
        image: The processed coin image (1000x1000)
        face_type: "obverse" or "reverse"
    
    Returns:
        numpy.ndarray: Rotated coin image
    """
    print(f"Rotating {face_type}")
    
    # load reference mask from MSD project team
    reference_mask = load_msd_reference_mask(face_type)
    if reference_mask is None:
        print("No reference mask available, returning original image")
        return image
    
    # find optimal rotation
    optimal_angle, optimal_score, scores, angles = find_optimal_rotation(image, reference_mask, face_type)
    
    # apply rotation
    if abs(optimal_angle) > 0.5:
        rotated_image = apply_rotation_normalization(image, optimal_angle)
        print(f"Applied rotation: {optimal_angle:.1f}°")
        return rotated_image
    else:
        print("No rotation needed")
        return image


def rotate_coin_pair(obverse_image, reverse_image):
    """
    Rotate both obverse and reverse MSD coin images respectively to optimal orientations.
    
    Args:
        obverse_image: The processed obverse coin image (1000x1000)
        reverse_image: The processed reverse coin image (1000x1000)
    
    Returns:
        tuple: (rotated_obverse, rotated_reverse) - Both rotated coin images
    """
    print("MSD rotation normalization")
    rotated_obverse = rotate_coin(obverse_image, "obverse")
    rotated_reverse = rotate_coin(reverse_image, "reverse")
    print("\nRotation complete")
    return rotated_obverse, rotated_reverse