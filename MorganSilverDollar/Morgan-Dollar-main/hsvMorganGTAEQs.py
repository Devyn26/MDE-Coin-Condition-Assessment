'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 3/05/2025
'''
import cv2 as cv2
import os
import numpy as np
import statistics
import math
import time

# import ImageOpener
# from ImageOpener import loadImages
# import imutils
# from sklearn.decomposition import PCA
# from GUI import *

"""Calculates the HSV of Morgan Dollar Coins using the equations provided

The equations were provided by the GTA's original team and by the powerpoint slide given by Dr. Jones. The main purpose
of this file is to compare it to hsvMorganOpenCV.py and hsvMorganSmithEQ to see if there is any difference in the HSV calculated and to see
which one does the calculations faster.
"""

cyan = '\033[96m'
endCyan = '\033[0m'


# Returns the Mean, Median, and the Standard Deviation of the HSV values. Use output of Calculate_HSV() for HSV_Value
def mean_median_stdev_value(HSV_Value, scale):
    if scale > 1 or scale < 0:
        print("Invalid scale")
        return None, None
    # Presumably, all three lists within HSV_Value should be the same length
    HSV_length = math.floor(len(HSV_Value[0]) * scale)

    # gets the mean value
    mean_H = statistics.mean(HSV_Value[0][:HSV_length])
    mean_S = statistics.mean(HSV_Value[1][:HSV_length])
    mean_V = statistics.mean(HSV_Value[2][:HSV_length])

    # gets the median value
    median_H = statistics.median(HSV_Value[0][:HSV_length])
    median_S = statistics.median(HSV_Value[1][:HSV_length])
    median_V = statistics.median(HSV_Value[2][:HSV_length])
    #
    # # gets the standard deviation
    stdev_H = statistics.stdev(HSV_Value[0][:HSV_length])
    stdev_S = statistics.stdev(HSV_Value[1][:HSV_length])
    stdev_V = statistics.stdev(HSV_Value[2][:HSV_length])

    return [median_H, median_S, median_V], [mean_H, mean_S, mean_V], [stdev_H, stdev_S, stdev_V]


# Returns a 3D array of the Hue, Saturation, and Value of an image that is BGR. Use mask_and_in_paint_image() before
# using this so that background is not calculated in the HSV result.
def Calculate_HSV(image):
    """
     * In the function below RGB is converted to HSV using a set of formulas.  The
     * HSV values for each pixel within the image is calculated starting from (x=0, y=0) to
     * (x=width-1, y=height-1)
     *
     * Param:   Image - resulting RGB matrix obtained from cv2.imread()
     *
     * Return:  HSV_value - A list of 3 lists, the first list contains all hue values, the second contains all
     *                      saturation values, and the third contains all value values.
    """
    B, G, R = cv2.split(image)
    B, G, R = np.array(B.astype(np.float32)), np.array(G.astype(np.float32)), np.array(R.astype(np.float32))
    condition = (B != 0) | (G != 0) | (R != 0)

    B, G, R = np.extract(condition, B), np.extract(condition, G), np.extract(condition, R)
    # colorMin = np.min(imageFloatBGR, axis=0)
    colorMin = np.minimum.reduce([B, G, R])

    # colorMax = np.max(imageFloatBGR, axis=0)
    colorMax = np.maximum.reduce([B, G, R])
    colorRange = np.subtract(colorMax, colorMin)
    # H_prime = np.divide(np.add(G, B), C_range)
    """
     * Just as a note: 
     *    - Hue is the angle on the color wheel
     *    - Saturation is the distance from the center of the color wheel (further is greater saturation)
     *    - Value is the depth within the color wheel
    """
    # If there is not max or min, then the calculations are unnecessary
    zeroMatrix = np.zeros(colorMax.shape)

    redHighHue = np.divide(np.subtract(G, B), colorRange, out=np.full_like(R, 567.89), where=(colorRange != 0))
    greenHighHue = np.add(
        np.divide(np.subtract(B, R), colorRange, out=np.full_like(R, 567.89), where=(colorRange != 0)), 2.0)
    blueHighHue = np.add(np.divide(np.subtract(R, G), colorRange, out=np.full_like(R, 567.89), where=(colorRange != 0)),
                         4.0)

    hueConditions = [colorMax == colorMin, colorMax == R, colorMax == G, colorMax == B]

    hueOptions = [zeroMatrix, redHighHue, greenHighHue, blueHighHue]
    hsvHCalc = np.select(hueConditions, hueOptions, 567.89)
    maxValHue = np.max(hsvHCalc)
    print('\t', "The maximum value for Hue is", maxValHue)
    if maxValHue == 567.89:
        print("\n\nError!\n\n")
    hsvHueDegree = np.round(np.where(hsvHCalc == 0, 0, np.where(hsvHCalc < 0, np.add(np.multiply(hsvHCalc, 60), 360),
                                                                 np.multiply(hsvHCalc, 60))), 3)

    colorMaxNormalized = np.divide(colorMax, 255)
    colorRangeNormalized = np.divide(colorRange, 255)
    # Find the value, the value is determined by the color with the greatest strength in RGB
    # V = round(C_high_bin * 100, 3)
    V = np.round(np.multiply(colorMaxNormalized, 100), 3)
    # Find the saturation by applying a conversion formula, if all RGB values are the same then there is no sat
    Saturation = np.where(colorMax == colorMin, 0,
                          np.round(np.multiply(np.divide(colorRangeNormalized, colorMaxNormalized), 100), 3))
    HSV_val = np.array([hsvHueDegree, Saturation, V])
    return HSV_val


# Masks the image provided
def mask_and_no_in_paint_image(image):
    # Gets the width & height of the image
    h, w = image.shape[:2]

    # Radius includes the entire coin except for the edge ridges
    radius = 460  # THIS ISN'T CONSISTENT!

    # Initialize a new mask as an empty image
    maskCircle = np.zeros_like(image)

    # Gets half height and width of the image
    h_floor, w_floor = h // 2, w // 2

    # Create a circular mask to apply to each image
    maskCircle = np.array(cv2.circle(maskCircle, (h_floor, w_floor), radius, (255, 255, 255), -1))

    # Applies the circular mask to the image
    maskedImage = np.array(cv2.bitwise_and(image, maskCircle))

    return maskedImage


# I think this masks the image first and then tries to remove glare from the coin since the Morgan Dollar coin is so
# reflexive this actually alters / removes some of the features and detail of the coin. The input should be a BGR
# image, not a HSV image. I've added an additional parameter "glare" that can be used to change how much this is
# altered. The higher "glare" is, the more glare will be removed from the coin giving a more accurate color, however
# distortions will occur on the coin and detail will also be lost. This definitely matters for the conditioning team,
# but I'm unsure how much this matters for the color/toning team.
def mask_and_in_paint_image(image, glare):
    print("Using a threshold value of", glare, "to remove glare from Morgan Dollar Coin")
    # Gets the width & height of the image
    h, w = image.shape[:2]
    print(h, w)
    # print("height is ", h, "and width is", w)
    # Radius includes the entire coin except for the edge ridges
    radius = 460
    # Initialize a new mask as an empty image
    maskCircle = np.zeros_like(image)
    # Gets half width and height of the image
    h_floor, w_floor = h // 2, w // 2
    # Create a circular mask to apply to each image
    maskCircle = np.array(cv2.circle(maskCircle, (h_floor, w_floor), radius, (255, 255, 255), -1))
    # Applies the circular mask to the image
    maskedImage = np.array(cv2.bitwise_and(image, maskCircle))
    """
     * Image In-painting
    """

    """
        Applies a binary threshold to the image on the range of glare-255, and returns the threshold value and the 
        modified image
        Parameters:
            
            glare   -       threshold value
            255     -       the maximum value assigned to pixel exceeding threshold
            cv2.THRESH_BINARY   -       makes threshold() use the following algorithm:
                                                if pixel less than or equal to glare then set pixel to zero
                                                else set pixel to 255 (the third parameter given)
        Output:
            a list; list[0] holds the threshold value used, list[1] holds the modified image
    """

    return maskedImage


# def parseDatabaseHelper(path, glare):
#     excelName = 'Coin_Data.xlsx'
#     sheetName = 'Sheet1'
#
#     df = pd.read_excel(io=excelName, sheet_name=sheetName)
#     print("The first row of the input excel is:", df.columns[1], ",", df.columns[2], ",", df.columns[3], ",",
#           df.columns[4])
#     rows, columns = df.shape
#     print("The number of rows is", rows, "and the number of columns is", columns)
#
#     # List to store the Degree of Toning and other useful information
#     coinInfoList = {"Degree of Toning": [], "DLRC Inventory": [], "Name of Obverse Image": [],
#                     "Name of Reverse Image": [], "Obverse Mean of Hue": [], "Obverse Mean of Saturation": [],
#                     "Obverse Mean of Value": [], "Obverse Median of Hue": [], "Obverse Median of Saturation": [],
#                     "Obverse Median of Value": [], "Obverse Standard Deviation of Hue": [],
#                     "Obverse Standard Deviation of Saturation": [], "Obverse Standard Deviation of Value": [],
#                     "Reverse Mean of Hue": [], "Reverse Mean of Saturation": [],
#                     "Reverse Mean of Value": [], "Reverse Median of Hue": [], "Reverse Median of Saturation": [],
#                     "Reverse Median of Value": [], "Reverse Standard Deviation of Hue": [],
#                     "Reverse Standard Deviation of Saturation": [], "Reverse Standard Deviation of Value": []}
#
#     numInvalidFiles = 0
#     errorsSuffix = ""
#     numCoinsParsing = rows
#
#     print("There is are", numCoinsParsing + 1, "images to be parsed. Commencing parsing...")
#     for index in range(0, numCoinsParsing):
#         print("\n\n")
#         print("Parsing coin #:", index + 1)
#         print("Reading obverse image...")
#         imageFolderPath = path + "ScrapedImages/"
#         imgObverse = df.at[index, "Name of Obverse Image"] + ".jpg"
#         imgReverse = df.at[index, "Name of Reverse Image"] + ".jpg"
#
#         # file_path = os.path.abspath(imageFolderPath + imgObverse)
#         obverse_path = path + imgObverse
#         print("Obverse path is:", obverse_path)
#         if not os.path.isfile(obverse_path):
#             print("Error: " + obverse_path + " does not exist in the database")
#             numInvalidFiles = numInvalidFiles + 1
#             if numInvalidFiles < 5:
#                 errorsSuffix = errorsSuffix + "_e" + str(index + 1)
#             continue
#             # sys.exit("There is an error with the image " + file_path + " in the database")
#         coinInfoList["Degree of Toning"].append(df.at[index, "Degree of Toning"])
#         coinInfoList["DLRC Inventory"].append(df.at[index, "DLRC Inventory"])
#
#         coinInfoList["Name of Obverse Image"].append(imgObverse)
#         coinInfoList["Name of Reverse Image"].append(imgReverse)
#
#         imageBGR = np.array(cv2.imread(obverse_path))
#         # imageBGR = np.array(imageBGR)
#
#         print("Applying mask to obverse image...")
#         # maskedImage = mask_and_in_paint_image(imageBGR, glare)
#         maskedImage = determineTimeOfFunction2Arg(mask_and_in_paint_image, imageBGR, glare)
#
#         print("Calculating HSV of obverse image...")
#         # hsvOfCoin = Calculate_HSV(maskedImage)
#         hsvOfCoin = determineTimeOfFunction1Arg(Calculate_HSV, maskedImage)
#
#         print(
#             "Determining mean, median, and standard deviation of the Hue, Saturation, Value of the obverse image...")
#         # mean, median, stdDev = mean_median_stdev_value(hsvOfCoin, 1)
#         mean, median, stdDev = determineTimeOfFunction2Arg(mean_median_stdev_value, hsvOfCoin, 1)
#
#         coinInfoList["Obverse Mean of Hue"].append(mean[0])
#         coinInfoList["Obverse Mean of Saturation"].append(mean[1])
#         coinInfoList["Obverse Mean of Value"].append(mean[2])
#
#         coinInfoList["Obverse Median of Hue"].append(median[0])
#         coinInfoList["Obverse Median of Saturation"].append(median[1])
#         coinInfoList["Obverse Median of Value"].append(median[2])
#
#         coinInfoList["Obverse Standard Deviation of Hue"].append(stdDev[0])
#         coinInfoList["Obverse Standard Deviation of Saturation"].append(stdDev[1])
#         coinInfoList["Obverse Standard Deviation of Value"].append(stdDev[2])
#
#         print("Obverse calculations completed")
#
#         print("Reading reverse image...")
#         reverse_path = path + imgReverse
#         print("Reverse path is:", reverse_path)
#         if not os.path.isfile(reverse_path):
#             # print("There is an error with the image " + file_path + " in the database")
#             # continue
#             sys.exit("There is an error with the image " + reverse_path + " in the database")
#         imageBGR = np.array(cv2.imread(reverse_path))
#         # imageBGR = np.array(imageBGR)
#
#         print("Applying mask to reverse image...")
#         # maskedImage = mask_and_in_paint_image(imageBGR, glare)
#         maskedImage = determineTimeOfFunction2Arg(mask_and_in_paint_image, imageBGR, glare)
#
#         print("Calculating HSV of reverse image...")
#         # hsvOfCoin = Calculate_HSV(maskedImage)
#         hsvOfCoin = determineTimeOfFunction1Arg(Calculate_HSV, maskedImage)
#
#         print(
#             "Determining mean, median, and standard deviation of the Hue, Saturation, Value of the reverse image...")
#         # mean, median, stdDev = mean_median_stdev_value(hsvOfCoin, 1)
#         mean, median, stdDev = determineTimeOfFunction2Arg(mean_median_stdev_value, hsvOfCoin, 1)
#
#         coinInfoList["Reverse Mean of Hue"].append(mean[0])
#         coinInfoList["Reverse Mean of Saturation"].append(mean[1])
#         coinInfoList["Reverse Mean of Value"].append(mean[2])
#
#         coinInfoList["Reverse Median of Hue"].append(median[0])
#         coinInfoList["Reverse Median of Saturation"].append(median[1])
#         coinInfoList["Reverse Median of Value"].append(median[2])
#
#         coinInfoList["Reverse Standard Deviation of Hue"].append(stdDev[0])
#         coinInfoList["Reverse Standard Deviation of Saturation"].append(stdDev[1])
#         coinInfoList["Reverse Standard Deviation of Value"].append(stdDev[2])
#
#         print("Reverse calculations completed")
#
#     df = pd.DataFrame(coinInfoList,
#                       columns=["Degree of Toning", "DLRC Inventory", "Name of Obverse Image",
#                                "Name of Reverse Image", "Obverse Mean of Hue", "Obverse Mean of Saturation",
#                                "Obverse Mean of Value", "Obverse Median of Hue", "Obverse Median of Saturation",
#                                "Obverse Median of Value", "Obverse Standard Deviation of Hue",
#                                "Obverse Standard Deviation of Saturation", "Obverse Standard Deviation of Value",
#                                "Reverse Mean of Hue", "Reverse Mean of Saturation",
#                                "Reverse Mean of Value", "Reverse Median of Hue", "Reverse Median of Saturation",
#                                "Reverse Median of Value", "Reverse Standard Deviation of Hue",
#                                "Reverse Standard Deviation of Saturation", "Reverse Standard Deviation of Value"])
#     excelOutputName = 'Coin_Data_GTA_Calculations_' + 'Glare_' + str(glare) + '_Coins_' + str(numCoinsParsing)
#     print("Scraping was completed")
#     print("There was", numInvalidFiles, "invalid files")
#     if numInvalidFiles == 0:
#         print('There was', numInvalidFiles, 'invalid files. Scraping was successful')
#         errorsSuffix = '_Passed_' + str(numInvalidFiles) + '_errors' + '.xlsx'
#     else:
#         print('There was', numInvalidFiles, 'invalid files. Please check database for invalid or missing images')
#         if numInvalidFiles <= 5:
#             errorsSuffix = '_Failed_' + str(numInvalidFiles) + '_errors' + errorsSuffix + '.xlsx'
#         else:
#             errorsSuffix = '_Failed_' + str(numInvalidFiles) + '_errors' + '.xlsx'
#     excelOutputName = excelOutputName + errorsSuffix
#     df.to_excel(excelOutputName, sheet_name='Sheet1')

def determineTimeOfFunction1Arg(functionToTest, firstArg):
    startFunctionTime = time.time()

    returnOfFunc = functionToTest(firstArg)

    endFunctionTime = time.time()

    totalTimeOfFunction = round(endFunctionTime - startFunctionTime, 5)

    print("\t" + cyan, functionToTest.__name__, "took", str(totalTimeOfFunction), "seconds to run", endCyan)
    return returnOfFunc


def determineTimeOfFunction2Arg(functionToTest, firstArg, secArg):
    startFunctionTime = time.time()

    returnOfFunc = functionToTest(firstArg, secArg)

    endFunctionTime = time.time()

    totalTimeOfFunction = round((endFunctionTime - startFunctionTime), 5)

    print("\t" + cyan, functionToTest.__name__, "took", str(totalTimeOfFunction), "seconds to run", endCyan)
    return returnOfFunc


def determineTimeOfFunction4Arg(functionToTest, firstArg, secArg, thirdArg, fourthArg):
    startFunctionTime = time.time()

    returnOfFunc = functionToTest(firstArg, secArg, thirdArg, fourthArg)

    endFunctionTime = time.time()

    totalTimeOfFunction = round((endFunctionTime - startFunctionTime), 5)

    print("\t" + cyan, functionToTest.__name__, "took", str(totalTimeOfFunction), "seconds to run", endCyan)
    return returnOfFunc


def parseDatabase(databaseFolderPath, glare):
    # This will be used to represent the path to the folder that contains the database folder
    folderContainingDatabaseFolder = ""

    # Returns the absolute path to the directory/folder of this file
    currentFolder = os.path.dirname(__file__)

    print("This is the path of the file hsvMorganGTAEQs.py:", currentFolder)

    # Splits the path such that mainFolderAbs contains every but the last directory/folder and curFolder just
    # contains the last folder
    mainFolderAbs, curFolder = os.path.split(currentFolder)
    print(curFolder, "was removed from the string of the path")
    databasePathAbs = mainFolderAbs + "\\" + databaseFolderPath
    print("The images parsed by hsvMorganGTAEQs is the database located at:", databasePathAbs)
    print("Checking if this is a valid path...")

    if os.path.isdir(databasePathAbs):
        print("This is a valid path")
        # parseDatabaseHelper(databasePathAbs, glare)
        determineTimeOfFunction2Arg(parseDatabaseHelper, databasePathAbs, glare)
        return True
    else:
        print("Error the path is a not a valid directory")
        return False


def secToHMS(seconds):
    timeStr = ""
    if seconds >= 3600:
        hour = round((seconds / 3600.0), 0)
        timeStr = timeStr + str(hour) + " hrs"
    if seconds >= 60:
        minutesFloat = seconds / 60.0
        minute = round(minutesFloat, 2)
        timeStr = timeStr + str(minute) + " minutes"
        secondsRemain = round((minutesFloat - minute) * 60.0, 5)
        timeStr = timeStr + str(secondsRemain) + "seconds"
        return timeStr
    else:
        timeStr = str(round(seconds, 5)) + " seconds"
        return timeStr


# This functions only runs when the current file is executed as a script
if __name__ == '__main__':
    # This represent the folder containing the database
    # This path must be relative to the main folder
    mainStart = time.time()
    databaseFolder = "ScrapedImages/"

    # The second parameter used here is the glare/threshold that will be used on the image
    parseDatabase(databaseFolder, 255)
    mainEnd = time.time()
    totalTime = mainEnd - mainStart
    totalTimeStr = secToHMS(totalTime)
    print(cyan + "Parsing the entire database took", totalTimeStr, "" + endCyan)
