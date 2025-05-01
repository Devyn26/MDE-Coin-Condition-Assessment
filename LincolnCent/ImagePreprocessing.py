"""
Created by: John Anthony Kadian
Date: 4/29/2025

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
    Preprocessing Techniques used:
    - Grayscale and Gaussian blur
    - Hough–circle to find coin
    - Mask & crop to circle
    - Composite on white background and smooth edges
"""

def preprocess_coin_image_cv2(image, pad_frac=0.05):
    
    h, w = image.shape[:2]

    # grayscale and Gaussian blur
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2, sigmaY=2)

    # detect circle
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=h/8,
        param1=100,
        param2=30,
        minRadius=int(min(h, w)*0.3),
        maxRadius=int(min(h, w)*0.5)
    )

    # if no circle is detected, return None
    if circles is None:
        return None
    
    # get circle parameters
    x, y, r = np.round(circles[0][0]).astype(int)

    #. mask & crop to circle with smooth edges
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (x, y), r, 1.0, -1)
    # apply Gaussian blur to mask edges
    mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1, sigmaY=1)
    
    # crop to circle
    y1, y2 = max(y-r, 0), min(y+r, h)
    x1, x2 = max(x-r, 0), min(x+r, w)
    crop = image[y1:y2, x1:x2].astype(np.float32) / 255.0
    mask_crop = mask[y1:y2, x1:x2]

    # get crop dimensions for final padding
    h_c, w_c = crop.shape[:2]

    # compute bbox with padding
    ys, xs = np.where(mask_crop > 0.1)  # decreased threshold to include edge pixels
    if ys.size == 0 or xs.size == 0:
        return None
    

    y0, y1_ = ys.min(), ys.max()+1
    x0, x1_ = xs.min(), xs.max()+1

    # pad by pad_frac
    pad_y = int(pad_frac * (y1_ - y0))
    pad_x = int(pad_frac * (x1_ - x0))
    y0, x0 = max(y0-pad_y, 0), max(x0-pad_x, 0)
    y1_, x1_ = min(y1_+pad_y, h_c), min(x1_+pad_x, w_c)

    # composite on white background with smooth blending
    final_canvas = np.ones_like(crop)
    mask_crop_3ch = np.stack([mask_crop] * 3, axis=-1)
    final_canvas = crop * mask_crop_3ch + final_canvas * (1 - mask_crop_3ch)
    final = final_canvas[y0:y1_, x0:x1_]
    
    # convert back to uint8
    final = (final * 255).astype(np.uint8)
    return final


if __name__ == '__main__':
    # this is the image I'm using for local testing, change this to test other images
    
    
    orig_bgr = cv2.imread('LincolnCent/CentPhone.jpg')
    if orig_bgr is None:
        print("Error: CentPhone.jpg not found")
        exit(1)

    # coin-centric preprocess (circle crop + ellipse deskew)
    coin = preprocess_coin_image_cv2(orig_bgr)
    if coin is None:
        print("Failed to detect a coin in CentPhone.jpg")
        exit(1)

    # Convert to RGB for matplotlib
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(coin, cv2.COLOR_BGR2RGB)

    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(orig_rgb)
    ax1.set_title('Original CentPhone.jpg')
    ax1.axis('off')

    ax2.imshow(proc_rgb)
    ax2.set_title('Preprocessed')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()
