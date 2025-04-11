"""
ToningCoverage.py

This does the calculations for determining the toning coverage. What we did was we found the values of Saturation and
Value that was between the grayscale and the actual color. This then allowed us to find what pixels were toned (the
colors not on the grayscale) and what pixels were not toned (the pixels on the grayscale). We then took those pixels
and divided them by the overall pixels in the image allowing us to obtain a toning coverage percentage.

Author: Mathew Donlon, Zymmorrah Myers, Luis Vasquez Morales
Date: 4/27/2023
"""
import cv2 as cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import hsvMorganGTAEQs as eq
from PIL import Image
from time import perf_counter
import ToningMacros


# This will just print out to the console the degree of toning
def getToningCoverage(coin):
    #imageBGR = coin.colorimg
    #imageBGRMasked = eq.mask_and_no_in_paint_image(imageBGR)

    HSV_Value = eq.Calculate_HSV(coin.colorimg)

    toningCoverage = calcToningCoverage(HSV_Value, coin)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return toningCoverage
def calcToningCoverage(HSVImage, coin):
    # The following gets a 2 dimensional array of the values of the Hue, Saturation, and Value of the Image
    # print(HSVImage.shape)
    H, S, V = HSVImage
    H2, S2, V2 = HSVImage

    numNotCounted = 0

    # Makes each 2 dimensional array's value into float instead of integers and makes it a numpy array
    H2, S2, V2 = np.array(H2.astype(np.float32)), np.array(S2.astype(np.float32)), np.array(V2.astype(np.float32))
    H, S, V = np.array(H.astype(np.float32)), np.array(S.astype(np.float32)), np.array(V.astype(np.float32))
    print(H, S, V)
    # This condition is used so that only non-zero values are extracted from the numpy arrays
    condition = (H2 != 0) | (S2 != 0) | (V2 != 100)
    # Makes each numpy array only contain non-zero values
    H2, S2, V2 = np.extract(condition, H2), np.extract(condition, S2), np.extract(condition, V2)

    # Makes a zero matrix of the size of the hue matrix, they should all be the same size regardless though
    zeroMatrix = np.zeros(H.shape)

    # Makes a matrix the size of the hue matrix that contains all ones
    tonedMatrix = np.full(H.shape, 1)

    # This is the value that will be given to pixels that where not covered by the constrains, which means that they
    # are not toned and not toned.
    pixelsNotTonedValue = 0

    tonedConditions = \
        [V < 10, S >= 15, (V >= 10) & (V < 25)]

    # each hueConditions corresponds to the same index as hueOptions
    tonedOptions = [zeroMatrix, tonedMatrix, tonedMatrix]

    # Makes all pixels that fit the toned conditions 1, all that don't 0, and any that doesn't fit into either
    # condition is set to pixelsNotCovered (100)
    pixelsClassified = np.select(tonedConditions, tonedOptions, pixelsNotTonedValue)

    # Gets the coverage image
    getToningCoverageImage(coin, pixelsClassified)

    # The number of pixels there are
    numPixels = H2.size
    print(numPixels)


    # The number of non-zero pixels
    numTonedPixels = np.count_nonzero(pixelsClassified)

    # Converts the number of pixels and the number of toned pixels into floats, so that the percentage can be calculated
    numPixels, numTonedPixels = float(numPixels), float(numTonedPixels)

    # This is just used for error detection
    maxValHue = np.max(pixelsClassified)
    if maxValHue > 1:
        numNotCounted = (pixelsClassified == pixelsNotTonedValue).sum()
        numNotCounted = float(numNotCounted)
        percentNotCovered = (numNotCounted / numPixels) * 100.0
        strPerNotCovered = f"{percentNotCovered:.2f}%"
        strOutput = "The total not covered is " + str(numNotCounted) + " pixels. The total number of pixels are " + str(
            numPixels) + " pixels." + " The total percent not covered is " + strPerNotCovered
        print(
            "\n\nError!: One of the pixels was not classified as anything." + strOutput + " \n\n")

    #Added 15 to off set error
    toningCoverage = round((numTonedPixels / numPixels) * 100, 1)

    #if toningCoverage > 30:
        #toningCoverage = toningCoverage + 15
    #else:
        #toningCoverage = toningCoverage + 5
    print(toningCoverage)
    return toningCoverage, pixelsClassified


# Used to help determine the location of the coin image you want
def parseHelper(databasePath, ImagePath):
    # Returns the absolute path to the directory/folder of this file
    currentFolder = os.path.dirname(__file__)

    print("This is the path of the file ToningCoverage.py:", currentFolder)

    # Splits the path such that mainFolderAbs contains every but the last directory/folder and curFolder just
    # contains the last folder
    mainFolderAbs, curFolder = os.path.split(currentFolder)

    # print(curFolder, "was removed from the string of the path")
    databasePathAbs = mainFolderAbs + "\\" + databasePath

    print("The images parsed by hsvMorganGTAEQs is the database located at:", databasePathAbs)
    print("Checking if this is a valid path...")

    if os.path.isdir(databasePathAbs):
        print("This is a valid path")

        path = databasePathAbs
    else:
        print("Error the path is a not a valid directory")
        return False

    ImagePath = databasePathAbs + ImagePath

    # This is used to determine runtime during development, will be removed once everything is optimized
    # PrintCoinHSVToningCoverage(ImagePath)

    print("toning coverage was completed")


def getToningCoverageImage(coin, pixelsClassified):
    # Prepare images for modification
    pixelsCopy = pixelsClassified.copy()
    pixelsCopy = np.reshape(pixelsCopy, (1000, 1000))
    copyColorImg = coin.colorimg.copy()
    pixelsCopy = pixelsCopy.astype('uint8')
    # Inverse Binary image
    pixelsCopy[pixelsCopy == 0] = 255
    pixelsCopy[pixelsCopy == 1] = 0
    # Convert Gray to RGB
    coloredPixels = cv2.cvtColor(pixelsCopy, cv2.COLOR_GRAY2RGB)
    # Sets all black pixels to red
    coloredPixels[np.where((coloredPixels == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
    # Overlay coverage image over original image
    alpha, beta, gamma = 1, 0.4, -100
    silly = cv2.addWeighted(copyColorImg, alpha, coloredPixels, beta, gamma)
    # Save image
    path = os.path.abspath('Images/coverage') + '\\'
    cv2.imwrite(path + 'coverageImage.jpg', silly)

    return silly

# This functions only runs when the current file is executed as a script
if __name__ == '__main__':
    # This represent the folder containing the database
    # This path must be relative to the main folder
    databaseFolder = "ScrapedImages/"

    # image that will have it's toning coverage done
    imageChosen = "1878-CC_$1_NGC_AG-03_REVERSE_2421329.jpg"

    # Parses through each image in the database to display, no inpainting is used
    parseHelper(databaseFolder, imageChosen)
