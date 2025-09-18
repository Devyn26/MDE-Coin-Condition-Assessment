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
    Process a single image by isolating the imperfections and quantifying the white pixels in the canny image
    Based on: MorganSilverDollar/Morgan-Dollar-main/ConditionImperfectionEval.py process_imperfection_image()
    Original Author: Creed Jones, Modified By: Lizzie LaVallee
    Date: 8 Sep 2022, Modified: 10 Mar 2023
    
    Args:
        image: The coin image (1000x1000) with white background
    
    Returns:
        numpy.ndarray: Preprocessed Canny edge image ready for mask comparison
    """
    # convert to grayscale if the image is in color
    if len(image.shape) == 3: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # pre-processing
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    clImg = clahe.apply(gray)
    blurred = cv2.bilateralFilter(src=clImg, d=13, sigmaColor=200, sigmaSpace=200)  # bilateral filter
    cannyImg = cv2.Canny(blurred, 50, 150)
    
    return cannyImg


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
    
    print(f"Loaded {face_type} reference mask: {mask.shape}")
    return mask


def find_optimal_rotation(coin_image, reference_mask, face_type, angle_range=180, step=1.0):
    """
    Find the optimal rotation angle using template matching with normalized cross-correlation.
    This matches the MSD grading model's approach for template comparison.
    
    Args:
        coin_image: The processed coin image (1000x1000) with white background
        reference_mask: The reference mask (1000x1000)
        face_type: "obverse" or "reverse" for display purposes
        angle_range: Range of angles to test (default 180 degrees)
        step: Step size for angle testing (default 1.0 degrees)
    
    Returns:
        float: Optimal rotation angle in degrees
    """
    print(f"Finding optimal rotation for {face_type}")
    
    # preprocess coin image using MSD pipeline
    coin_edges = preprocess_for_msd_comparison(coin_image)
    print(f"Preprocessed coin using MSD Canny pipeline")
    print(f"Using template matching with normalized cross-correlation (MSD team)")
    
    best_angle = 0
    best_score = -1
    scores = []
    angles = []
    
    print(f"Testing rotation angles from -{angle_range/2}° to +{angle_range/2}° (step: {step}°)")
    
    test_angles = np.arange(-angle_range/2, angle_range/2 + step, step)
    
    for angle in test_angles:
        if abs(angle) < 0.1:  # Skip very small rotations
            rotated_edges = coin_edges
        else:
            # Rotate the preprocessed edges
            h, w = coin_edges.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_edges = cv2.warpAffine(coin_edges, rotation_matrix, (w, h), borderValue=0)
        

        # Based on: MorganSilverDollar/Morgan-Dollar-main/patternMatching.py patternMatch()
        result = cv2.matchTemplate(rotated_edges, reference_mask, cv2.TM_CCOEFF_NORMED)
        
        # Get the maximum correlation value
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        correlation_score = max_val
        
        scores.append(correlation_score)
        angles.append(angle)
        
        # minimum correlation for correct orientation
        if correlation_score < best_score or best_score == -1:
            best_score = correlation_score
            best_angle = angle
    
    print(f"Best rotation: {best_angle:.1f}° (min correlation: {best_score:.4f})")
    
    # display visualized results
    create_rotation_visualization(coin_image, coin_edges, reference_mask, best_angle, 
                                 best_score, face_type, angles, scores)
    
    return best_angle


# ChatGPT Generated function purely for visualization of the pipeline
# will be deleted later when integrating into the pipeline
def create_rotation_visualization(coin_image, coin_edges, reference_mask, best_angle, 
                                 best_score, face_type, angles, scores):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    # Row 1: Original images
    # Original coin
    coin_rgb = cv2.cvtColor(coin_image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(coin_rgb)
    axes[0, 0].set_title(f"Original {face_type.title()}\n(1000x1000, 97.5% diameter)")
    axes[0, 0].axis('off')
    
    # Preprocessed edges
    axes[0, 1].imshow(coin_edges, cmap='gray')
    axes[0, 1].set_title("MSD Preprocessed Edges\n(CLAHE + Bilateral + Canny)")
    axes[0, 1].axis('off')
    
    # Reference mask
    axes[0, 2].imshow(reference_mask, cmap='gray')
    axes[0, 2].set_title("Reference Mask\n(Regions of Interest)")
    axes[0, 2].axis('off')
    
    # Score plot
    axes[0, 3].plot(angles, scores, 'b-', linewidth=2)
    axes[0, 3].axvline(x=best_angle, color='r', linestyle='--', linewidth=2, label=f'Best: {best_angle:.1f}°')
    axes[0, 3].set_xlabel('Rotation Angle (degrees)')
    axes[0, 3].set_ylabel('Correlation Score')
    axes[0, 3].set_title('Rotation Score vs Angle')
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].legend()
    
    # Row 2: Rotation results
    # Best rotation edges
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
    axes[1, 1].set_title(f"Mask Applied to Rotated Edges")
    axes[1, 1].axis('off')
    
    # Final rotated coin
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
    
    # Comparison: Original vs Rotated
    comparison = np.hstack([coin_rgb, rotated_coin_rgb])
    axes[1, 3].imshow(comparison)
    axes[1, 3].set_title("Comparison\n(Original | Rotated)")
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()


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
    optimal_angle = find_optimal_rotation(image, reference_mask, face_type)
    
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
