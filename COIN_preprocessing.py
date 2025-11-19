"""
COIN_preprocessing.py

Preprocesses coin images (obverse and reverse) from raw "user-taken"
photo images of a coin into a 1000x1000 image with the coin centered, scaled properly, 
and rotated to the correct orientation.

Author: John Anthony Kadian
Date: 9/15/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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

def detect_coin(image, num_blurs=3, blur_kernel=(15, 15)):
    """
    Detect coin using the new preprocessing pipeline with configurable blur parameters.
    Uses the improved preprocessing from LWC_preprocessing_skimage.py
    """
    h, w = image.shape[:2]
    center = (w//2, h//2)
    assumed_radius = int(w * 0.4)  # 80% diameter = 40% radius
    
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # apply configurable number of blurs to suppress fine details
    current = gray.copy()
    for i in range(num_blurs):
        current = cv2.GaussianBlur(current, blur_kernel, 0)
    
    # apply bilateral filter
    bilateral = cv2.bilateralFilter(current, d=5, sigmaColor=50, sigmaSpace=50)
    
    # houghCircles on bilateral filtered image
    circles = cv2.HoughCircles(
        bilateral,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(assumed_radius * 0.4),
        param1=50,
        param2=30,
        minRadius=int(assumed_radius * 0.4),
        maxRadius=int(assumed_radius * 1.6)
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # take first circle from houghCircles
        if len(circles) > 0:
            x, y, r = circles[0]
            print(f"HoughCircles: center=({x}, {y}), radius={r}")
            return x, y, r, bilateral, bilateral
        else:
            print("No circles detected by HoughCircles")
            return None, None, None, bilateral, bilateral
    else:
        print("No circles detected by HoughCircles")
        return None, None, None, bilateral, bilateral

def get_coin_brightness(img, x, y, r):
    """
    Calculate the average brightness of a coin within a circular region.
    Useful for anomaly detection. Determines if a coin is too bright or dim for 
    accurate grading.
    Author: Eric Morley
    Args:
        img: The input image
        x, y: Center coordinates of the coin
        r: Radius of the coin
    
    Returns:
        float: Average brightness value (0-255)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # create circular mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, thickness=-1)
    
    # compute mean brightness inside the mask
    mean_val = cv2.mean(gray, mask=mask)[0]
    return mean_val

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
    # read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    # first crop and scale to1000x1000
    processed_image = smart_crop_and_scale(image)

    # reference mask for rotation normalization
    reference_mask = load_reference_mask(face_type)

    # detect coin circle using new preprocessing pipeline
    x, y, r, edges, edges_before_morph = detect_coin(processed_image)

    # visualize results for debugging
    debug_image = processed_image.copy()
    
    # assumed coin circle (green)
    cv2.circle(debug_image, (500, 500), 487, (0, 255, 0), 2)
    cv2.putText(debug_image, "ASSUMED COIN POSITION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if x is not None and y is not None and r is not None:
        # draw detected circle (red)
        cv2.circle(debug_image, (x, y), r, (0, 0, 255), 3)
        cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(debug_image, "DETECTED COIN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # calculate coin brightness
        brightness = get_coin_brightness(processed_image, x, y, r)
        print(f"Average brightness inside coin: {brightness:.2f}")
        
        # scale image to center the detected coin to 975px diameter
        output = center_and_scale_coin(processed_image, x, y, r)
        print(f"Final: detected center=({x}, {y}), radius={r} -> scaled to 97.5% (487px radius)")
    else:
        print("No circles were detected")
        output = processed_image

    '''
    # output plots
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    

    # top row
    ax1.imshow(original_rgb)
    ax1.set_title(f"Original Image ({image.shape[1]}x{image.shape[0]})", fontsize=10)
    ax1.axis("off")
    
    ax2.imshow(debug_rgb)
    ax2.set_title("Processed Image with Circle Detection", fontsize=10)
    ax2.axis("off")
    
    ax3.imshow(edges_before_morph, cmap='gray')
    ax3.set_title("Preprocessed Image (Hough Input)", fontsize=10)
    ax3.axis("off")
    
    # bottom row
    ax4.imshow(edges, cmap='gray')
    ax4.set_title("Preprocessed Image (Hough Input)", fontsize=10)
    ax4.axis("off")
    
    ax5.imshow(final_rgb)
    ax5.set_title("Final Coin Centered", fontsize=10)
    ax5.axis("off")
    
    # hide empty subplot
    ax6.axis("off")
    
    plt.tight_layout()
    plt.show()
    '''
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
    obverse_path = "msd-front-1921.jpg"
    reverse_path = "msd-back-1921.jpg"
    
    # process both obverse and reverse images
    obverse_result, reverse_result = process_coin(obverse_path, reverse_path)
    
    # save extracted coins
    cv2.imwrite('COIN_Proc_ob.jpg', obverse_result)
    cv2.imwrite('COIN_Proc_rev.jpg', reverse_result)
    
    print(f"\nExtraction Results:")
    print(f"Obverse extracted and saved to COIN_Proc_ob.jpg")
    print(f"Reverse extracted and saved to COIN_Proc_rev.jpg")
    
    # identify and apply rotation using both faces
    from COIN_identifier import identify_and_rotate_coin_pair
    rotated_obverse, rotated_reverse, obv_type, rev_type = identify_and_rotate_coin_pair(
        obverse_result, reverse_result)

    # save rotated finals
    cv2.imwrite('COIN_Final_ob.jpg', rotated_obverse)
    cv2.imwrite('COIN_Final_rev.jpg', rotated_reverse)

    print("\nFinal Results:")
    print(f"Obverse: {obv_type} -> saved to COIN_Final_ob.jpg")
    print(f"Reverse: {rev_type} -> saved to COIN_Final_rev.jpg")
