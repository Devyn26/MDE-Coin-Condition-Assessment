"""
MSD_preprocessing.py

Preprocesses Morgan Silver Dollar coin images (obverse and reverse) from raw "user-taken"
photo images of a coin into a 1000x1000 image with the coin centered, scaled properly, 
and rotated to the correct orientation.

Author: John Anthony Kadian
Date: 9/15/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from MSD_rotation import rotate_coin_pair

def smart_crop_and_scale(image):
    """
    Assume coin is placed within a circle with diameter 80% of width, centered. 
    Crop square and scale to 1000x1000 to speed up coin detection and increase accuracy. (more focused search space)
    This is designed assuming our GUI will have a stencil of roughly where the coin should
    be placed in the user's photo.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The processed square and resized image with assumed coin circle.
    """
    h, w = image.shape[:2]
    
    # assumed coin circle (80% of width, centered)
    coin_diameter = int(w * 0.8)
    coin_radius = coin_diameter // 2
    center_x, center_y = w // 2, h // 2
    
    # crop square around the assumed coin circle
    # square size = 100% of width (or height, whichever is smaller)
    crop_size = min(w, h)
    half_crop = crop_size // 2
    
    x1 = max(0, center_x - half_crop)
    y1 = max(0, center_y - half_crop)
    x2 = min(w, center_x + half_crop)
    y2 = min(h, center_y + half_crop)
    
    # crop the image to the square
    cropped = image[y1:y2, x1:x2]
    
    # scale to 1000x1000, with respect to the assumed coin circle
    resized = cv2.resize(cropped, (1000, 1000))
    
    return resized

def detect_coin(image):
    """
    Detect coin using edge detection within the assumed 80% radius.
    Uses HoughCircles to find circular patterns from fragmented edges.
    """
    h, w = image.shape[:2]
    center = (w//2, h//2)
    assumed_radius = int(w * 0.4)  # 80% diameter = 40% radius
    
    # create mask for assumed coin circle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, assumed_radius, 255, -1)
    
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # canny edge detection with higher thresholds to reduce noise
    edges = cv2.Canny(blurred, 100, 200)
    
    # apply mask to edges
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # clean up edges with morphological operations
    # idea from https://www.reddit.com/r/computervision/comments/1k9p83h/detecting_striped_circles_using_computer_vision/
    # to clean up the outer edges of the coin, to overcome complex grooves and ridges
    kernel = np.ones((3,3), np.uint8)
    cleaned_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel)
    cleaned_edges = cv2.morphologyEx(cleaned_edges, cv2.MORPH_OPEN, kernel)
    
    # using HoughCircles method to find circular patterns from fragmented edges
    circles = cv2.HoughCircles(cleaned_edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                              param1=50, param2=20, minRadius=int(assumed_radius*0.6), 
                              maxRadius=int(assumed_radius*1.2))
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # take first circle from HoughCircles (highest confidence)
        if len(circles) > 0:
            x, y, r = circles[0]
            # expand radius by 5% because the inner rim is almost always the HoughCircles circle with the highest confidence
            expanded_r = int(r * 1.05)
            print(f"HoughCircles: center=({x}, {y}), radius={r} -> {expanded_r}")
            return x, y, expanded_r, cleaned_edges
        
        # old center-based selection - it used the same circle for every test case I tried
        # but this method is slower and likely less accurate than using cv2.HoughCircles built sorting
        # best_circle = None
        # min_distance = float('inf')
        # 
        # for circle in circles:
        #     x, y, r = circle
        #     # Calculate distance from center
        #     distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        #     if distance < min_distance:
        #         min_distance = distance
        #         best_circle = circle
        # 
        # if best_circle is not None:
        #     x, y, r = best_circle
        #     # Expand radius by 5% to get outer rim instead of inner rim
        #     expanded_r = int(r * 1.05)
        #     print(f"HoughCircles: center=({x}, {y}), radius={r} -> {expanded_r}")
        #     return x, y, expanded_r, cleaned_edges
    
    print("No coin detected")
    return None, None, None, cleaned_edges

def center_and_scale_coin(image, coin_x, coin_y, coin_radius):
    """
    Extract the coin from detected location and center it in a 1000x1000 image
    with the coin taking up exactly 975px diameter and white background.

    * The 975px diameter was determined by measuring the average coin diameter in the test images in the MSD
    grading model image dataset *
    
    Args:
        image: The 1000x1000 processed image
        coin_x, coin_y, coin_radius: Detected coin parameters
    
    Returns:
        Final 1000x1000 image with coin centered at 97.5% diameter and white background
    """
    # target: 975px diameter = 487.5px radius (round down because pixels are integers)
    target_radius = 487
    target_center = (500, 500)
    
    # determine scale factor to make coin exactly target_radius
    scale_factor = target_radius / coin_radius
    
    # new coin position parameters after scaling
    new_coin_x = int(coin_x * scale_factor)
    new_coin_y = int(coin_y * scale_factor)
    
    # now scale the entire image
    new_size = int(1000 * scale_factor)
    scaled_image = cv2.resize(image, (new_size, new_size))
    
    # offset to center the coin
    offset_x = target_center[0] - new_coin_x
    offset_y = target_center[1] - new_coin_y
    
    # create final image canvas starting with 1000x1000 white background
    final_canvas = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    
    # paste coordinates - where to place the coin on the 1000x1000 canvas
    paste_x1 = max(0, offset_x)
    paste_y1 = max(0, offset_y)
    paste_x2 = min(1000, offset_x + new_size)
    paste_y2 = min(1000, offset_y + new_size)
    
    # source coordinates - the part of the scaled image to copy from
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = src_x1 + (paste_x2 - paste_x1)
    src_y2 = src_y1 + (paste_y2 - paste_y1)
    
    # paste coin in the white 1000x1000 canvas
    final_canvas[paste_y1:paste_y2, paste_x1:paste_x2] = scaled_image[src_y1:src_y2, src_x1:src_x2]
    
    # create and apply circular mask for cropping background outside the coin region
    final_mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.circle(final_mask, target_center, target_radius, 255, -1)
    final_canvas[final_mask == 0] = [255, 255, 255]
    
    return final_canvas

def load_reference_mask(face_type):
    """
    Load the appropriate reference mask for rotation normalization.
    
    Args:
        face_type (str): Either "obverse" or "reverse"
    
    Returns:
        numpy.ndarray: The reference mask image (1000x1000)
    """
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

def process_coin_image(image_path, face_type="obverse"):
    """
    Processes a coin image: crops to square, scales to 1000x1000, detects coin circle,
    and creates a properly centered and cropped final image.

    Args:
        image_path (str): The path to the image file.
        face_type (str): Either "obverse" or "reverse" to determine which mask to use.

    Returns:
        numpy.ndarray: The processed coin image (1000x1000) with coin centered and background white.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    # first crop and scale to1000x1000
    processed_image = smart_crop_and_scale(image)

    # reference mask for rotation normalization from MSD project team
    reference_mask = load_reference_mask(face_type)

    # detect coin circle using HoughCircles detection
    x, y, r, edges = detect_coin(processed_image)

    # visualize results for debugging and visual feedback
    debug_image = processed_image.copy()
    
    # assumed coin circle (green) - 97.5% of 1000px = 487px radius scaled from 80% of original image width
    cv2.circle(debug_image, (500, 500), 487, (0, 255, 0), 2)
    cv2.putText(debug_image, "ASSUMED COIN POSITION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if x is not None and y is not None and r is not None:
        # draw detected circle (red)
        cv2.circle(debug_image, (x, y), r, (0, 0, 255), 3)
        cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(debug_image, "DETECTED COIN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # now scale image to to center the DETECTED coin to 975px diameter, instead of the assumed coin circle
        output = center_and_scale_coin(processed_image, x, y, r)
        print(f"Final: detected center=({x}, {y}), radius={r} -> scaled to 97.5% (487px radius)")
    else:
        print("No circles were detected.")
        output = processed_image

    # output plots
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
    

    ax1.imshow(original_rgb)
    ax1.set_title(f"Original Image ({image.shape[1]}x{image.shape[0]})")
    ax1.axis("off")
    

    ax2.imshow(debug_rgb)
    ax2.set_title("Processed Image (1000x1000) with Circle Detection")
    ax2.axis("off")
    

    ax3.imshow(edges, cmap='gray')
    ax3.set_title("Edge Detection Within Assumed Radius")
    ax3.axis("off")
    
    ax4.imshow(final_rgb)
    ax4.set_title("Final Coin 1000x1000 and centered")
    ax4.axis("off")
    
    plt.tight_layout()
    plt.show()

    return output

def process_coin(obverse_path, reverse_path):
    """
    Processes both obverse and reverse coin images using the same pipeline
    but with appropriate masks for each face type.

    Args:
        obverse_path (str): Path to the obverse (front) image file.
        reverse_path (str): Path to the reverse (back) image file.

    Returns:
        tuple: (processed_obverse, processed_reverse) - Both 1000x1000 images
    """
    print("Processing obverse (front)...")
    obverse_result = process_coin_image(obverse_path, "obverse")
    
    print("\nProcessing reverse (back)...")
    reverse_result = process_coin_image(reverse_path, "reverse")
    
    return obverse_result, reverse_result

if __name__ == "__main__":
    obverse_path = "front-old.jpg"
    reverse_path = "back-old.jpg"
    
    # process both obverse and reverse images respectively
    obverse_result, reverse_result = process_coin(obverse_path, reverse_path)
    
    # apply MSD rotation normalization
    print("\nApplying MSD rotation normalization...")
    rotated_obverse, rotated_reverse = rotate_coin_pair(obverse_result, reverse_result)
    
    # save the results
    cv2.imwrite('MSD_Proc_ob.jpg', rotated_obverse)
    cv2.imwrite('MSD_Proc_rev.jpg', rotated_reverse)