"""
ConditionImperfectionEval.py

Evaluates the imperfections of the coin in the field. Apply a custom mask to read only what is
supposed to be the flat areas of the coin for scratches and dirt. Quantify the imperfections using Canny image and
number of white pixels, or Sobel image and average value of white pixels.

Original Author: Creed Jones
Date: 8 Sep 2022
Modified By: Lizzie LaVallee
Date: 10 Mar 2023
"""
import numpy as np
import pandas as pd
import os
import cv2

from CoinImage import CoinImage
from MaskImage import MaskImage
from plotConditionData import scatterPlot
import EdgeEval
from buildDatabase import compileConditionData


def process_imperfection_image(coin, mask):
    """
        Process a single image by isolating the imperfections and quantifying the white pixels in the canny image
        coinimage: the coin object
        mask: the custom mask file to apply
        return: the masked preprocessed image
    """

    # pre-processing
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    clImg = clahe.apply(coin.grayimg)
    blurred = cv2.bilateralFilter(src=clImg, d=13, sigmaColor=200, sigmaSpace=200)  # bilateral filter
    dy, dx = EdgeEval.sobelEdge(blurred)
    dx = dx.astype(np.uint8)
    dy = dy.astype(np.uint8)
    sobelImg = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    cannyImg = EdgeEval.cannyEdge(blurred, 50, 150)
    # apply the mask on the edge read images
    masked_c = MaskImage.custom_mask(cannyImg, mask)
    masked_s = MaskImage.custom_mask(sobelImg, mask)
    width = sobelImg.shape[0]
    height = sobelImg.shape[1]
    totalValue_s = 0
    numPixels_s = 0
    features_c = 0
    # find total number of white canny pixels, and avg value of sobel non-black pixels
    # for w in range(width):
    #     for h in range(height):
    #         pixel_s = masked_s[w, h]
    #         pixel_c = masked_c[w, h]
    #         if pixel_s > 0:
    #             numPixels_s += 1
    #             totalValue_s += pixel_s
    #         if pixel_c > 0:
    #             features_c += 1

    return masked_c, np.sum(masked_c == 255)  # masked image for pdf, canny score


def measureImperfections(dirname, maskname, face):
    """ Produce a dataset of the grade vs imperfections """
    colnames = ['densities', 'grades']
    df = pd.DataFrame(columns=colnames)
    fileList = os.listdir(dirname)
    densities_c, grades = np.zeros(shape=(len(fileList), 1)), np.zeros(shape=(len(fileList), 1))
    for index, filename in enumerate(fileList):
        coinInfo = filename.split(' ')
        if coinInfo.count('') > 0:
            coinInfo.remove('')
        if coinInfo[-1][:-4] != face:
            continue
        c = CoinImage()
        c.load(dirname + filename)
        df.loc[filename] = filename

        # densities_s[index], densities_c[index] = process_imperfection_image(c, maskname)
        images, densities_c = process_imperfection_image(c, maskname)

        # Add the density of this coin to the database csv file. Choose between canny/sobel (_c/_s)
        compileConditionData(densities_c, filename, False, 'EdgeFreq Flat')

    # scatterPlot(x=densities_s, y=grades, files=fileList, title="Imperfection Quantification Using Average Sobel Value",
    #             xlabel='Avg Value')  # plot average value of sobel
    # scatterPlot(x=densities_c, y=grades, files=fileList,
    #             title="Imperfection Quantification Using Num Canny White Pixels",
    #             xlabel='Num White Pixels')  # plot number of white pixels of canny
    return densities_c, None


if __name__ == '__main__':
    obv_dir = os.path.abspath('ScrapedImages/obverse') + '\\'
    rev_dir = os.path.abspath('ScrapedImages/reverse') + '\\'
    obv_mask = os.path.abspath('CustomMasks/') + '\\' + 'obv_flat.jpg'
    rev_mask = os.path.abspath('CustomMasks/') + '\\' + 'rev_flat.jpg'
    measureImperfections(obv_dir, obv_mask, 'obverse')
    print('finished obv')
    measureImperfections(rev_dir, rev_mask, 'reverse')

