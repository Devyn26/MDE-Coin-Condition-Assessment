"""
COIN Rotation Normalization

Neutral rotation system that tests both LWC and MSD masks to determine
the best fit, providing both rotation and coin identification.

Author: John Anthony Kadian
Date: 10/12/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def preprocess_for_coin_comparison(image):
    """
    Neutral preprocessing pipeline (mirrors both LWC and MSD):
    - Gray -> Gaussian blur (sigma=1.5)
    - Sobel X/Y (ksize=5)
    - Gradient magnitude normalized to 0..255
    - Light median filter
    - Circle mask to coin region
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
    """Load LWC reference mask for the given face type."""
    mask_dir = os.path.join('LincolnCent', 'CustomMasks')
    if face_type.lower() == 'obverse':
        primary = os.path.join(mask_dir, 'obv_head_inv.jpg')
        fallback = os.path.join(mask_dir, 'lwc_obv_flat_inv.jpg')
    elif face_type.lower() == 'reverse':
        primary = os.path.join(mask_dir, 'rev_wheat_stalks_inv.jpg')
        fallback = os.path.join(mask_dir, 'lwc_rev_flat_inv.jpg')
    else:
        raise ValueError("face_type must be 'obverse' or 'reverse'")

    mask = cv2.imread(primary, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = cv2.imread(fallback, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load LWC mask for {face_type}")
        return None
    # Invert mask per updated requirement (use inverse of existing assets)
    mask = cv2.bitwise_not(mask)
    print(f"Loaded LWC {face_type} reference mask (inverted): {mask.shape}")
    return mask


def load_msd_reference_mask(face_type):
    """Load MSD reference mask for the given face type."""
    mask_dir = "MorganSilverDollar/Morgan-Dollar-main/CustomMasks"
    
    if face_type.lower() == "obverse":
        mask_path = os.path.join(mask_dir, "obv_flat.jpg")
    elif face_type.lower() == "reverse":
        mask_path = os.path.join(mask_dir, "rev_flat.jpg")
    else:
        raise ValueError("face_type must be 'obverse' or 'reverse'")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load MSD mask {mask_path}")
        return None
    # Invert mask per updated requirement
    mask = cv2.bitwise_not(mask)
    print(f"Loaded MSD {face_type} reference mask (inverted): {mask.shape}")
    return mask


def normalize_correlation_score(correlation_score, reference_mask):
    """
    Normalize correlation score based on mask characteristics.
    Options:
    1. Mask area normalization: divide by active mask area
    2. Mask complexity normalization: account for mask fragmentation
    3. Mask density normalization: account for mask density distribution
    """
    # Calculate mask statistics
    mask_area = np.sum(reference_mask > 0)  # Number of non-zero pixels
    mask_density = mask_area / (reference_mask.shape[0] * reference_mask.shape[1])  # Density
    
    # Normalize by mask area (larger masks tend to have higher absolute correlations)
    area_normalized_score = correlation_score / (mask_area / 1000000)  # Scale by area
    
    # Alternative: normalize by mask density
    density_normalized_score = correlation_score / mask_density
    
    # Return the area-normalized score (you can experiment with density_normalized_score)
    return area_normalized_score, mask_area, mask_density


def find_optimal_rotation_with_mask(coin_image, reference_mask, face_type, coin_type, angle_range=180, step=1.0):
    """
    Find optimal rotation using a specific mask type with normalized correlation scores.
    """
    print(f"Testing {coin_type} {face_type} mask")
    
    coin_edges = preprocess_for_coin_comparison(coin_image)
    
    best_angle = 0
    best_score = -1
    best_raw_score = -1
    scores = []
    raw_scores = []
    angles = []
    
    test_angles = np.arange(-angle_range/2, angle_range/2 + step, step)
    
    for angle in test_angles:
        if abs(angle) < 0.1:
            rotated_edges = coin_edges
        else:
            h, w = coin_edges.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_edges = cv2.warpAffine(coin_edges, rotation_matrix, (w, h), borderValue=0)
        
        result = cv2.matchTemplate(rotated_edges, reference_mask, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        raw_correlation_score = max_val
        
        # Normalize the correlation score
        normalized_score, mask_area, mask_density = normalize_correlation_score(raw_correlation_score, reference_mask)
        
        scores.append(normalized_score)
        raw_scores.append(raw_correlation_score)
        angles.append(angle)
        
        if normalized_score > best_score or best_score == -1:
            best_score = normalized_score
            best_raw_score = raw_correlation_score
            best_angle = angle
    
    print(f"{coin_type} {face_type}: Best rotation: {best_angle:.1f}° (raw: {best_raw_score:.4f}, normalized: {best_score:.4f})")
    print(f"  Mask area: {mask_area}, density: {mask_density:.4f}")
    return best_angle, best_score, scores, angles


def find_optimal_rotation_for_identification(coin_image, face_type, angle_range=180, step=1.0):
    """
    Test both LWC and MSD masks to find the best fit for identification only.
    Returns the identified coin type and its raw rotation angle.
    """
    print(f"Finding optimal rotation for {face_type} (testing both LWC and MSD)")
    
    # Load both mask types
    lwc_mask = load_lwc_reference_mask(face_type)
    msd_mask = load_msd_reference_mask(face_type)
    
    if lwc_mask is None and msd_mask is None:
        print("No reference masks available, returning original image")
        return 0, "unknown"
    
    best_overall_angle = 0
    best_overall_score = -1
    best_coin_type = "unknown"
    best_scores = []
    best_angles = []
    
    # Initialize variables for visualization
    lwc_angle, lwc_score, lwc_scores, lwc_angles = 0, -1, [], []
    msd_angle, msd_score, msd_scores, msd_angles = 0, -1, [], []
    
    # Test LWC mask
    if lwc_mask is not None:
        lwc_angle, lwc_score, lwc_scores, lwc_angles = find_optimal_rotation_with_mask(
            coin_image, lwc_mask, face_type, "LWC", angle_range, step)
        
        if lwc_score > best_overall_score or best_overall_score == -1:
            best_overall_score = lwc_score
            best_overall_angle = lwc_angle
            best_coin_type = "LWC"
            best_scores = lwc_scores
            best_angles = lwc_angles
    
    # Test MSD mask
    if msd_mask is not None:
        msd_angle, msd_score, msd_scores, msd_angles = find_optimal_rotation_with_mask(
            coin_image, msd_mask, face_type, "MSD", angle_range, step)
        
        if msd_score > best_overall_score or best_overall_score == -1:
            best_overall_score = msd_score
            best_overall_angle = msd_angle
            best_coin_type = "MSD"
            best_scores = msd_scores
            best_angles = msd_angles
    
    print(f"Best match: {best_coin_type} {face_type} (rotation: {best_overall_angle:.1f}°, correlation: {best_overall_score:.4f})")
    
    # Create comprehensive visualization
    create_rotation_visualization(coin_image, face_type, best_coin_type, best_overall_angle, 
                                 best_overall_score, best_angles, best_scores,
                                 lwc_angle, lwc_score, lwc_scores, lwc_angles,
                                 msd_angle, msd_score, msd_scores, msd_angles)
    
    return best_overall_angle, best_coin_type


def find_optimal_rotation_standard(coin_image, reference_mask, face_type, angle_range=180, step=1.0):
    """
    Standard rotation finding without normalization (for actual rotation).
    """
    print(f"Finding optimal rotation for {face_type} (standard method)")
    
    coin_edges = preprocess_for_coin_comparison(coin_image)
    
    best_angle = 0
    best_score = -1
    scores = []
    angles = []
    
    test_angles = np.arange(-angle_range/2, angle_range/2 + step, step)
    
    for angle in test_angles:
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
    
    print(f"Standard rotation: {best_angle:.1f}° (max correlation: {best_score:.4f})")
    return best_angle


def find_optimal_rotation(coin_image, face_type, angle_range=180, step=1.0):
    """
    Two-step process: 1) Identify coin type using normalized scores, 2) Find rotation using standard method.
    """
    # Step 1: Identify coin type using normalized comparison
    identified_angle, coin_type = find_optimal_rotation_for_identification(coin_image, face_type, angle_range, step)
    
    # Step 2: Find actual rotation using standard method with the identified mask
    if coin_type == "LWC":
        reference_mask = load_lwc_reference_mask(face_type)
    elif coin_type == "MSD":
        reference_mask = load_msd_reference_mask(face_type)
    else:
        print(f"Unknown coin type: {coin_type}, using no rotation")
        return 0, coin_type
    
    if reference_mask is None:
        print(f"No reference mask available for {coin_type}, using no rotation")
        return 0, coin_type
    
    # Use standard rotation method for the actual rotation
    actual_angle = find_optimal_rotation_standard(coin_image, reference_mask, face_type, angle_range, step)
    
    return actual_angle, coin_type


def create_rotation_visualization(coin_image, face_type, coin_type, best_angle, 
                                 best_score, angles, scores, lwc_angle, lwc_score, 
                                 lwc_scores, lwc_angles, msd_angle, msd_score, 
                                 msd_scores, msd_angles):
    """Create comprehensive visualization showing both LWC and MSD test results."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: Original and preprocessing
    coin_rgb = cv2.cvtColor(coin_image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(coin_rgb)
    axes[0, 0].set_title(f"Original {face_type.title()}\n(1000x1000, 97.5% diameter)")
    axes[0, 0].axis('off')
    
    coin_edges = preprocess_for_coin_comparison(coin_image)
    axes[0, 1].imshow(coin_edges, cmap='gray')
    axes[0, 1].set_title("Preprocessed Edges\n(Sobel magnitude)")
    axes[0, 1].axis('off')
    
    # LWC vs MSD comparison plot
    axes[0, 2].plot(lwc_angles, lwc_scores, 'b-', linewidth=2, label='LWC', alpha=0.7)
    axes[0, 2].plot(msd_angles, msd_scores, 'r-', linewidth=2, label='MSD', alpha=0.7)
    axes[0, 2].axvline(x=lwc_angle, color='b', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 2].axvline(x=msd_angle, color='r', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 2].set_xlabel('Rotation Angle (degrees)')
    axes[0, 2].set_ylabel('Correlation Score')
    axes[0, 2].set_title('LWC vs MSD Rotation Scores')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Best result summary
    axes[0, 3].text(0.1, 0.8, f"LWC {face_type}:", fontsize=12, weight='bold')
    axes[0, 3].text(0.1, 0.7, f"  Best angle: {lwc_angle:.1f}°", fontsize=10)
    axes[0, 3].text(0.1, 0.6, f"  Score: {lwc_score:.4f}", fontsize=10)
    axes[0, 3].text(0.1, 0.4, f"MSD {face_type}:", fontsize=12, weight='bold')
    axes[0, 3].text(0.1, 0.3, f"  Best angle: {msd_angle:.1f}°", fontsize=10)
    axes[0, 3].text(0.1, 0.2, f"  Score: {msd_score:.4f}", fontsize=10)
    axes[0, 3].text(0.1, 0.05, f"Selected: {coin_type}", fontsize=14, weight='bold', 
                   color='green' if coin_type == 'LWC' else 'red')
    axes[0, 3].set_xlim(0, 1)
    axes[0, 3].set_ylim(0, 1)
    axes[0, 3].axis('off')
    
    # Row 2: LWC results
    if abs(lwc_angle) > 0.1:
        h, w = coin_edges.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, lwc_angle, 1.0)
        lwc_rotated_edges = cv2.warpAffine(coin_edges, rotation_matrix, (w, h), borderValue=0)
    else:
        lwc_rotated_edges = coin_edges
    
    axes[1, 0].imshow(lwc_rotated_edges, cmap='gray')
    axes[1, 0].set_title(f"LWC Rotated Edges\n({lwc_angle:.1f}°)")
    axes[1, 0].axis('off')
    
    # LWC mask
    lwc_mask = load_lwc_reference_mask(face_type)
    if lwc_mask is not None:
        axes[1, 1].imshow(lwc_mask, cmap='gray')
        axes[1, 1].set_title("LWC Reference Mask")
        axes[1, 1].axis('off')
    
    # LWC final result
    if abs(lwc_angle) > 0.1:
        h, w = coin_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, lwc_angle, 1.0)
        lwc_rotated_coin = cv2.warpAffine(coin_image, rotation_matrix, (w, h), borderValue=(255, 255, 255))
    else:
        lwc_rotated_coin = coin_image
    
    lwc_rotated_coin_rgb = cv2.cvtColor(lwc_rotated_coin, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(lwc_rotated_coin_rgb)
    axes[1, 2].set_title(f"LWC Final Result\n({lwc_angle:.1f}°)")
    axes[1, 2].axis('off')
    
    # LWC comparison
    lwc_comparison = np.hstack([coin_rgb, lwc_rotated_coin_rgb])
    axes[1, 3].imshow(lwc_comparison)
    axes[1, 3].set_title("LWC Comparison\n(Original | Rotated)")
    axes[1, 3].axis('off')
    
    # Row 3: MSD results
    if abs(msd_angle) > 0.1:
        h, w = coin_edges.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, msd_angle, 1.0)
        msd_rotated_edges = cv2.warpAffine(coin_edges, rotation_matrix, (w, h), borderValue=0)
    else:
        msd_rotated_edges = coin_edges
    
    axes[2, 0].imshow(msd_rotated_edges, cmap='gray')
    axes[2, 0].set_title(f"MSD Rotated Edges\n({msd_angle:.1f}°)")
    axes[2, 0].axis('off')
    
    # MSD mask
    msd_mask = load_msd_reference_mask(face_type)
    if msd_mask is not None:
        axes[2, 1].imshow(msd_mask, cmap='gray')
        axes[2, 1].set_title("MSD Reference Mask")
        axes[2, 1].axis('off')
    
    # MSD final result
    if abs(msd_angle) > 0.1:
        h, w = coin_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, msd_angle, 1.0)
        msd_rotated_coin = cv2.warpAffine(coin_image, rotation_matrix, (w, h), borderValue=(255, 255, 255))
    else:
        msd_rotated_coin = coin_image
    
    msd_rotated_coin_rgb = cv2.cvtColor(msd_rotated_coin, cv2.COLOR_BGR2RGB)
    axes[2, 2].imshow(msd_rotated_coin_rgb)
    axes[2, 2].set_title(f"MSD Final Result\n({msd_angle:.1f}°)")
    axes[2, 2].axis('off')
    
    # MSD comparison
    msd_comparison = np.hstack([coin_rgb, msd_rotated_coin_rgb])
    axes[2, 3].imshow(msd_comparison)
    axes[2, 3].set_title("MSD Comparison\n(Original | Rotated)")
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.show()


def apply_rotation_normalization(coin_image, angle):
    """Apply rotation normalization to the coin image."""
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
    """
    Rotate a coin using the best matching mask (LWC or MSD).
    Returns the rotated image and identified coin type.
    """
    print(f"Rotating {face_type} (testing both LWC and MSD masks)")
    
    optimal_angle, coin_type = find_optimal_rotation(image, face_type)
    
    if abs(optimal_angle) > 0.5:
        rotated_image = apply_rotation_normalization(image, optimal_angle)
        print(f"Applied rotation: {optimal_angle:.1f}° (identified as {coin_type})")
        return rotated_image, coin_type
    else:
        print(f"No rotation needed (identified as {coin_type})")
        return image, coin_type


def rotate_coin_pair(obverse_image, reverse_image):
    """
    Rotate both obverse and reverse coin images using the best matching masks.
    Returns rotated images and identified coin types.
    """
    print("COIN rotation normalization (testing LWC and MSD)")
    
    rotated_obverse, obverse_type = rotate_coin(obverse_image, "obverse")
    rotated_reverse, reverse_type = rotate_coin(reverse_image, "reverse")
    
    print(f"\nRotation complete:")
    print(f"Obverse identified as: {obverse_type}")
    print(f"Reverse identified as: {reverse_type}")
    
    return rotated_obverse, rotated_reverse, obverse_type, reverse_type
