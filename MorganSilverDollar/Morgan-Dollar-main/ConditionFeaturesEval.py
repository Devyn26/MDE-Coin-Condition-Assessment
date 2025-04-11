"""
ConditionFeaturesEval.py

~

Original Author: Creed Jones
Date: 8 Sep 2022
Modified By: Jasper Emick, Lizzie LaVallee
Date: 10 Mar 2023
"""
import numpy as np
import pandas as pd
import os
import cv2

from CoinImage import CoinImage
from MaskImage import MaskImage
import EdgeEval
from buildDatabase import compileConditionData


def process_one_image(coinimg, wedges):
    M = MaskImage()
    #coinfeat = np.zeros((1, 0), dtype=float)
    quadDen = np.zeros(len(wedges))
    for wedgect in range(len(wedges)):
        M.clear()

        if len(wedges) == 1:
            wdgRad = 0
        else:
            wdgRad = coinimg.coinradius
        M.create_wedge(start_angle=wedges[wedgect][0], end_angle=wedges[wedgect][1], wedgeRadius=wdgRad,
                       xc=int(coinimg.coincenter[0]), yc=int(coinimg.coincenter[1]))

        # feat = np.asarray(coinimg.applymask(M)).reshape(1, -1)
        # coinfeat = np.concatenate((coinfeat, feat), axis=1)

        den = EdgeEval.getEdgeReading(coin=coinimg, mask=M)
        quadDen[wedgect] = den

    return quadDen


def get_features_training(doWedge):
    """ Obtains the density of physical features found across the surface of a coin"""

    paths = [os.path.abspath('ScrapedImages/obverse') + '\\', os.path.abspath('ScrapedImages/reverse') + '\\']
    fileLists = [os.listdir(paths[0]), os.listdir(paths[1])]
    I = CoinImage()
    colNames = []

    if doWedge:
        wedges = ((0, 90), (90, 180), (180, 270), (270, 0))
    else:
        wedges = ((0, 360),)

    for wedgect in range(len(wedges)):
        colNames.extend(I.featurenames("W" + str(wedgect)))

    # Dataframe for gradient features
    df = pd.DataFrame(columns=colNames)

    # Iterate through obverse and reverse
    displayMap = {0: 'Obverse', 1: 'Reverse'}
    for side, fileList in enumerate(fileLists):

        for index, filename in enumerate(fileList):

            print(displayMap[side] + " " + str(index))
            nameArr = filename.split()
            if nameArr[-1] != 'obverse.jpg' and nameArr[-1] != 'reverse.jpg':
                continue

            I.load(paths[side] + filename)
            I.filename = filename
            if I.findcenter() is None:
                print("Broken Coin" + str(filename))
                continue

            # Actual condition evaluation, returns feature density and pre-defined grade
            den = process_one_image(coinimg=I, wedges=wedges)
            # If properly evaluated, pixel count (den) is added to an array, else skip to next image
            if den is None:
                continue

            # Writes evaluated condition data to csv database
            #compileConditionData(density=den, filename=filename, quadrant=doWedge)

    # plot regression model
    # scatterPlot(x=densities, y=grades, files=fileList, title="Feature Density", xlabel="Density")
    writeDir = os.path.abspath('starter_kit_mostly_fixed/') + '\\'
    df.to_excel(writeDir + 'features.xlsx')


def process_one_image_custom_mask(coinimg, mask):
    img, densities_c = EdgeEval.getEdgeReading(coin=coinimg, mask=mask, isCustom=True)
    return img, densities_c


def get_features_training_custom_masks(obv_dir, rev_dir):
    """ Obtains the density of physical features found across the surface of a coin"""
    # custom masks
    obv_redorange = os.path.abspath('CustomMasks/') + '\\' + 'obv_red_orange_mask.jpg'
    rev_redorange = os.path.abspath('CustomMasks/') + '\\' + 'rev_red_orange_mask.jpg'
    obv_yellow = os.path.abspath('CustomMasks/') + '\\' + 'obv_yellow_mask.jpg'
    rev_yellow = os.path.abspath('CustomMasks/') + '\\' + 'rev_yellow_mask.jpg'
    obv_green = os.path.abspath('CustomMasks/') + '\\' + 'obv_green_mask.jpg'
    rev_green = os.path.abspath('CustomMasks/') + '\\' + 'rev_green_mask.jpg'

    obv = {'RedOrange': obv_redorange, 'Yellow': obv_yellow, 'Green': obv_green}
    rev = {'RedOrange': rev_redorange, 'Yellow': rev_yellow, 'Green': rev_green}

    if obv_dir is not None:
        obv_files = os.listdir(obv_dir)
        for index, filename in enumerate(obv_files):
            for colname in obv.keys():
                coinInfo = filename.split(' ')
                if coinInfo.count('') > 0:
                    coinInfo.remove('')
                if coinInfo[-1][:-4] != 'obverse':
                    continue
                c = CoinImage()
                c.load(obv_dir + filename)

                densities_c = process_one_image_custom_mask(c, obv[colname])

                # Add the density of this coin to the database csv file. Choose between canny/sobel (_c/_s)
                compileConditionData(densities_c, filename, False, 'EdgeFreq ' + colname)
    if rev_dir is not None:
        rev_files = os.listdir(rev_dir)
        for index, filename in enumerate(rev_files):
            for colname in rev.keys():
                coinInfo = filename.split(' ')
                if coinInfo.count('') > 0:
                    coinInfo.remove('')
                if coinInfo[-1][:-4] != 'reverse':
                    continue
                c = CoinImage()
                c.load(rev_dir + filename)

                densities_c = process_one_image_custom_mask(c, rev[colname])

                # Add the density of this coin to the database csv file. Choose between canny/sobel (_c/_s)
                compileConditionData(densities_c, filename, False, 'EdgeFreq ' + colname)

def getConditionImage(coin):
    return EdgeEval.getEdgeReading(coin, None, False)[0]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   obv_dir = os.path.abspath('ScrapedImages/obverse') + '\\'
   rev_dir = os.path.abspath('ScrapedImages/reverse') + '\\'
   get_features_training_custom_masks(obv_dir, rev_dir)
    #get_features_training(False)
