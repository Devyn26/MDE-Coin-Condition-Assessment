"""
EdgeEval.py

Used for determining the edge frequency of each coin, also contains preprocessing methods for edge detection

Author: Jasper Emick
Date: 10 Mar 2023

Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 3/05/2025
"""
import numpy as np
import cv2
from skimage import morphology
from matplotlib import pyplot as plt


def getEdgeReading(coin, mask, isCustom=True):

    # Gets the human grade for the image
    # fileList = coin.filename.split(" ")
    # if fileList.count('') != 0:
    #     fileList.remove('')
    # DLGrade = "".join([c for c in fileList[3] if c.isnumeric()])
    # if DLGrade == '' or int(DLGrade) > 70:
    #     return None

    # Pre-processing
    equalized = histEqualization(coin.grayimg)

    edges = evaluateEdge(equalized, coin)

    if isCustom is True:
        # MaskImage.custom_mask(edges, mask) idk why this isnt working so i copied the function below:
        mask = cv2.imread(mask, 0)
        masked_img = cv2.bitwise_and(mask, edges)
        edges_masked = masked_img
    else:
        edges_masked = edges

    # cv2.bitwise_and(mask.M, edges)
    # cv2.namedWindow('sdfsd', cv2.WINDOW_NORMAL)
    # cv2.imshow("sdfsd", edges_masked)
    # cv2.waitKey(0)
    # Count white pixels
    white_pixels = np.sum(edges_masked == 255)

    return edges_masked, white_pixels


def evaluateEdge(img, coin):
    """
    Blurs the image using a bilateral filter, then obtains the sobel gradient magnitude of the image with image
    morphology to clean up any irrelevant noise in the image.
    """
    blurred = cv2.bilateralFilter(src=img, d=9, sigmaColor=200, sigmaSpace=200)

    # Sobel Edge reading
    dx, dy = sobelEdge(blurred)

    # Converts sobel images to binary gradient magnitude
    magnitude = np.sqrt(np.add(np.multiply(dx, dx), np.multiply(dy, dy)))
    maxi = 0.22 * np.max(magnitude[:])
    magnitude[magnitude > maxi] = 255
    magnitude[magnitude <= maxi] = 0

    # Image morphology - groups of connected white pixels which fall below the set threshold in size will be removed
    # from the image; this is employed to clean up remaining extraneous noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    morph = cv2.morphologyEx(magnitude, cv2.MORPH_OPEN, k)
    morph = morph.astype('uint8') > 0
    saved = morphology.remove_small_objects(morph, min_size=100)

    # Save and convert Morphed image
    saved = saved.astype('uint8') * 255

    return saved


def sobelEdge(blur):
    """
    Obtains the Sobel edge reading of an already blurred image
    """
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    dx = abs_grad_x.astype(np.float32)
    dy = abs_grad_y.astype(np.float32)

    return dx, dy


def cannyEdge(blur, low, high):
    canny = cv2.Canny(blur, low, high)
    return canny


def histEqualization(img):
    """
    Applies an advanced from of histogram equalization onto an image (Contrast Limited Adaptive Histogram Equalization)
    """
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    clImg = clahe.apply(img)

    return clImg


def displayImages(blurred, coin, saved):

    fig, ax = plt.subplots(1, 3)
    cv2.imshow("wewef", blurred)
    ax[0].imshow(cv2.cvtColor(coin.colorimg, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original", size=12)
    ax[0].axis('off')
    ax[1].imshow(blurred, cmap='gray')
    ax[1].set_title("Preprocessing", size=12)
    ax[1].axis('off')
    ax[2].imshow(saved, cmap='gray')
    ax[2].set_title("Edge Filtered", size=12)
    ax[2].axis('off')
    fig.show()
    cv2.waitKey(0)
