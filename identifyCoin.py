import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from COIN_rotation import (
    preprocess_for_coin_comparison,
    load_lwc_reference_mask,
    load_msd_reference_mask,
    normalize_correlation_score
)
from COIN_preprocessing import smart_crop_and_scale, detect_coin, center_and_scale_coin


def identify_coin_type(coin_image, face_type, angle_range=180, step=1.0):
    """
    Identify coin type (LWC or MSD) using normalized correlation scores.
    Uses the exact same logic as COIN_rotation.py for consistency.
    """
    print(f"Identifying {face_type} coin type...")
    
    # Load both mask types
    lwc_mask = load_lwc_reference_mask(face_type)
    msd_mask = load_msd_reference_mask(face_type)
    
    if lwc_mask is None and msd_mask is None:
        print("No reference masks available")
        return "unknown", -1
    
    best_overall_angle = 0
    best_overall_score = -1
    best_coin_type = "unknown"
    
    # Test LWC mask using the same method as COIN_rotation.py
    if lwc_mask is not None:
        lwc_angle, lwc_score, lwc_scores, lwc_angles = find_optimal_rotation_with_mask(
            coin_image, lwc_mask, face_type, "LWC", angle_range, step)
        
        if lwc_score > best_overall_score or best_overall_score == -1:
            best_overall_score = lwc_score
            best_overall_angle = lwc_angle
            best_coin_type = "LWC"
    
    # Test MSD mask using the same method as COIN_rotation.py
    if msd_mask is not None:
        msd_angle, msd_score, msd_scores, msd_angles = find_optimal_rotation_with_mask(
            coin_image, msd_mask, face_type, "MSD", angle_range, step)
        
        if msd_score > best_overall_score or best_overall_score == -1:
            best_overall_score = msd_score
            best_overall_angle = msd_angle
            best_coin_type = "MSD"
    
    print(f"Identified as: {best_coin_type} (score: {best_overall_score:.4f})")
    return best_coin_type, best_overall_score


def test_mask_fast(coin_image, reference_mask, coin_type):
    """
    Fast mask testing by only checking 4 key angles: 0°, 90°, 180°, 270°.
    """
    coin_edges = preprocess_for_coin_comparison(coin_image)
    
    best_score = -1
    best_raw_score = -1
    test_angles = [0, 90, 180, 270]
    
    for angle in test_angles:
        if angle == 0:
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
        
        if normalized_score > best_score or best_score == -1:
            best_score = normalized_score
            best_raw_score = raw_correlation_score
    
    print(f"  {coin_type}: raw={best_raw_score:.4f}, normalized={best_score:.4f}, area={mask_area}, density={mask_density:.4f}")
    return best_score


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
        
        if normalized_score < best_score or best_score == -1:
            best_score = normalized_score
            best_raw_score = raw_correlation_score
            best_angle = angle
    
    print(f"{coin_type} {face_type}: Best rotation: {best_angle:.1f}° (raw: {best_raw_score:.4f}, normalized: {best_score:.4f})")
    print(f"  Mask area: {mask_area}, density: {mask_density:.4f}")
    return best_angle, best_score, scores, angles


def process_all_images_in_directory(directory_path="images"):
    """
    Process all images in the directory and create a collage showing coin identification results.
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found")
        return
    
    # Get all image files
    image_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        print(f"\nProcessing: {image_file}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load {image_file}")
            continue
        
        # Determine face type from directory structure
        if "front" in image_file.lower() or "obverse" in image_file.lower():
            face_type = "obverse"
        elif "back" in image_file.lower() or "reverse" in image_file.lower():
            face_type = "reverse"
        else:
            # Try to determine from directory path
            if "front" in directory_path.lower():
                face_type = "obverse"
            elif "back" in directory_path.lower():
                face_type = "reverse"
            else:
                face_type = "obverse"  # Default to obverse
        
        # Preprocess the image (extract and scale to 1000x1000)
        print(f"  Preprocessing {image_file}...")
        try:
            # First crop and scale to 1000x1000
            resized_image = smart_crop_and_scale(image)
            
            # Then detect and extract the coin
            x, y, r, bilateral1, bilateral2 = detect_coin(resized_image)
            
            if x is None or y is None or r is None:
                print(f"  Could not detect coin in {image_file}, skipping...")
                continue
                
            center = (x, y)
            radius = r
            print(f"  Coin detected: center=({center[0]:.1f}, {center[1]:.1f}), radius={radius:.1f}")
            
            # Create the final centered and scaled coin image
            detected_coin = center_and_scale_coin(resized_image, x, y, r)
            
        except Exception as e:
            print(f"  Error preprocessing {image_file}: {e}")
            continue
        
        # Identify coin type using the preprocessed coin
        coin_type, score = identify_coin_type(detected_coin, face_type)
        
        # Store results
        results.append({
            'filename': image_file,
            'image': detected_coin,  # Use the preprocessed coin for display
            'face_type': face_type,
            'coin_type': coin_type,
            'score': score
        })
    
    # Create collage
    create_identification_collage(results)


def create_identification_collage(results):
    """
    Create a collage showing all images with their identification results.
    """
    if not results:
        print("No results to display")
        return
    
    num_images = len(results)
    
    # Calculate grid dimensions (square-ish)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = axes.flatten()
    
    # Color mapping for coin types (normalized to 0-1 range)
    color_map = {
        'LWC': (0, 1, 0),        # Green
        'MSD': (0, 0, 1),        # Blue
        'unknown': (0.5, 0.5, 0.5)  # Gray
    }
    
    for i, result in enumerate(results):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Convert BGR to RGB for display
        display_image = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
        
        # Show image
        ax.imshow(display_image)
        ax.axis('off')
        
        # Add identification text
        coin_type = result['coin_type']
        score = result['score']
        face_type = result['face_type']
        
        # Create text with background
        text = f"{coin_type}\n{face_type}\n{score:.3f}"
        
        # Add text background
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                color=color_map.get(coin_type, (0, 0, 0)))
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Coin Identification Results ({len(results)} images)", fontsize=16, y=0.98)
    plt.show()


if __name__ == "__main__":
    # Test on all images in the images directory
    print("Processing images from images/front and images/back directories...")
    
    # Process front directory
    if os.path.exists("images/front"):
        print("\n=== PROCESSING FRONT/OBVERSE IMAGES ===")
        process_all_images_in_directory("images/front")
    
    # Process back directory  
    if os.path.exists("images/back"):
        print("\n=== PROCESSING BACK/REVERSE IMAGES ===")
        process_all_images_in_directory("images/back")
    
    # Also process root images directory if it exists
    if os.path.exists("images") and not os.path.exists("images/front") and not os.path.exists("images/back"):
        print("\n=== PROCESSING ROOT IMAGES DIRECTORY ===")
        process_all_images_in_directory("images")
