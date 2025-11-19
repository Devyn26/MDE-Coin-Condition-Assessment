"""
pipe.py

Preprocesses Morgan Silver Dollar coin images (obverse and reverse) from raw "user-taken"
photo images of a coin into a 1000x1000 image with the coin centered, scaled properly, 
and rotated to the correct orientation.

Checks brightness/glare and brightens the coin if needed.

Author: John Anthony Kadian, Eric Morley
Date: 10/15/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from MSD_rotation import rotate_coin_pair
from MorganSilverDollar.Morgan_Dollar_main import InputCoin
from LincolnCent import ImageHSV, patternMatching, WheatStalkGrader
from COIN_preprocessing import smart_crop_and_scale, center_and_scale_coin, load_reference_mask, process_coin_image, process_coin
import time
from PIL import Image

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

            if not (brightness_glare_check(image, x, y, r)):
                return None, None, None, bilateral, bilateral

            return x, y, r, bilateral, bilateral
        else:
            print("No circles detected by HoughCircles")
            return None, None, None, bilateral, bilateral
    else:
        print("No circles detected by HoughCircles")
        return None, None, None, bilateral, bilateral

def brightness_glare_check(img: np.ndarray, x: int, y: int, r: int) -> int:
    """
    Checks if the coin is too bright/dark or if there is too much glare for grading.
    """
    bright_thresh = 215 # upper threshold for average coin brightness
    dark_thresh = 80 # lower threshold for average coin brightness
    glare_thresh = 3 # threshold percentage for amount >= 250 pixels in coin

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coin_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(coin_mask, (x, y), r, 255, -1)

    # glare check setup
    coin_pixels = gray[coin_mask == 255]

    brightness = cv2.mean(gray, mask=coin_mask)[0]
    print(f"Average brightness inside coin: {brightness:.2f}")

    glare_pct = np.sum(coin_pixels >= 255) / coin_pixels.size * 100
    print(f"Glare pixels >= 255: {glare_pct:.2f}%")

    # check for coin being too bright/dark
    if (brightness > bright_thresh):
        print("Coin is too bright")
        return 0
    elif (brightness < dark_thresh):
        print("Coin is too dark")
        return 0

    # glare check
    if glare_pct > glare_thresh:
        print("Too much glare detected on coin")
        return 0

    return 1

def brighten_coin(img: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    """
    Brightens the coin so its average brightness approaches target_brightness.
    Stops scaling if the brightest coin pixel reaches 255.
    """
    target_brightness = 180

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coin_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(coin_mask, (x, y), r, 255, -1)

    coin_pixels = gray[coin_mask == 255]

    current_mean = np.mean(coin_pixels)
    current_max = np.max(coin_pixels)

    print(f"Before brightening mean: {current_mean:.2f}, Max: {current_max}")

    if current_mean >= target_brightness or current_max == 255:
        print("Coin already bright enough")
        return img

    scale = target_brightness / current_mean

    max_allowed_scale = 255.0 / current_max
    scale = min(scale, max_allowed_scale)

    print(f"Applying scale factor: {scale:.3f}")

    brightened = img.copy()
    for c in range(3):
        channel = brightened[:, :, c].astype(np.float32)
        channel[coin_mask == 255] *= scale
        channel = np.clip(channel, 0, 255)
        brightened[:, :, c] = channel.astype(np.uint8)

    new_gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
    new_pixels = new_gray[coin_mask == 255]
    print(f"After brightening - Mean: {np.mean(new_pixels):.2f}, Max: {np.max(new_pixels)}")

    return brightened

def runPre(obverse_path, reverse_path):
    obverse_result, reverse_result = process_coin(obverse_path, reverse_path)

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

    if obv_type == 'MSD' and rev_type == 'MSD':
        print("running MSD")
        InputCoin.runMSDCode(rotated_obverse, rotated_reverse)
    elif obv_type == 'LWC' and rev_type == 'LWC':
        print("running LWC")
        #color_label, color_outlier = ImageHSV.ONLY_ONE_COIN_INPUT_FOR_COLOR_CLASSIFICATION(rotated_obverse)
        #is_brown = (str(color_label).strip().lower() == "brown")
        print("started timer")
        start_time = time.perf_counter()
        fm_grade_float = float(patternMatching.gradeCoin(rotated_obverse, False, True))
        fm_grade = round(fm_grade_float)
        left_grade, right_grade, wheat_oss = WheatStalkGrader.gradeWheatStalkPenny(rotated_reverse)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")
        grade_wheat = round(wheat_oss)
        grade = (fm_grade + grade_wheat) / 2
        print(f"Grade: {grade:.2f}")
        manualGen(rotated_obverse, rotated_reverse, grade, 1,
                  fm_grade_float=fm_grade_float,
                  left_score=left_grade, right_score=right_grade,
                  wheat_sheldon=wheat_oss, elapsed_seconds=elapsed)
        
def manualGen(obv_img, rev_img, grade, coin=0, *, fm_grade_float=None, left_score=None, right_score=None, wheat_sheldon=None, elapsed_seconds=None):
    obv_img = cv2.cvtColor(obv_img, cv2.COLOR_BGR2RGB)
    rev_img = cv2.cvtColor(rev_img, cv2.COLOR_BGR2RGB)

    c = InputCoin.inputCoin(coin_type="Lincoln Wheat Cent")
    dr = c.detailedResults

    dr.ogObverse = Image.fromarray(obv_img)
    dr.ogReverse = Image.fromarray(rev_img)

    dr.flatObverse = None
    dr.flatReverse = None

    dr.condMasks["highSigObverse"] = None
    dr.condMasks["highSigReverse"] = None
    dr.condMasks["lowSigObverse"] = None
    dr.condMasks["lowSigReverse"] = None
    dr.condMasks["rimObverse"] = None
    dr.condMasks["rimReverse"] = None

    dr.conditionObverse = None
    dr.conditionReverse = None

    dr.conditionScore = grade
    dr.brillianceScore = None
    dr.histBrilliance = None

    # Attach LWC-specific metrics for the tailored report
    dr.lwc_metrics = {
        "featureMatchGrade": fm_grade_float if fm_grade_float is not None else grade,
        "wheatLeftScore": left_score,
        "wheatRightScore": right_score,
        "wheatSheldon": wheat_sheldon,
        "combinedGrade": grade,
        "elapsedSeconds": elapsed_seconds,
    }

    print("Manual Report Generation Started")

    c.generateDetailedResults()

if __name__ == "__main__":
    obverse_path = "front-noisy.jpg"
    reverse_path = "back-noisy.jpg"
    
    # process both obverse and reverse images respectively
    obverse_result, reverse_result = process_coin(obverse_path, reverse_path)
    
    # apply MSD rotation normalization
    print("\nApplying MSD rotation normalization...")
    rotated_obverse, rotated_reverse = rotate_coin_pair(obverse_result, reverse_result)
    
    # save the results
    cv2.imwrite('MSD_Proc_ob-old.jpg', rotated_obverse)
    cv2.imwrite('MSD_Proc_rev-old.jpg', rotated_reverse)

    ob = cv2.imread('MSD_Proc_ob.jpg')
    rev = cv2.imread('MSD_Proc_rev.jpg')

    InputCoin.runMSDCode(rotated_obverse, rotated_reverse)