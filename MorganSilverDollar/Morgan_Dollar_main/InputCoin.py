"""
InputCoin.py

~

Author: Jasper Emick, Lizzie LaVallee
Date: 10 Mar 2023

Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 3/05/2025

F25-06: 
- Still need to remove toning code
- Still has deprecated code to remove
- Still some unused library calls
"""
import numpy as np
import os

from . import hsvMorganGTAEQs as eq
from .CoinImage import CoinImage
from .ConditionFeaturesEval import process_one_image_custom_mask, getConditionImage
from .ToningProcessing import getToningScore
from .ToningCoverage import getToningCoverageImage
from .ToningCoverage import calcToningCoverage
from .ConditionImperfectionEval import process_imperfection_image
from .Brilliance import getBrilliance_And_Percent_Silver, getBrillianceHist
from .MorganGrader import Grader
from .DetailedResults import PDF, generateTemplate

import cv2
from PIL import Image


class inputCoin:

    def __init__(self):
        # Objects
        self.grader = Grader()
        self.obverseCoin = CoinImage()  # CoinImage object
        self.reverseCoin = CoinImage()  # CoinImage object
        self.detailedResults = PDF()
        # Features
        self.predictedGrade = None  # Int value
        self.obverseFeatures = []  # List in order: Flat Mask -> Face Mask -> Hair Mask -> Rim Mask
        self.reverseFeatures = []  # List in order: Flat Mask -> Center Mask -> Wings Mask -> Rim Mask
        self.obverseBrilliance = None  # Float value
        self.reverseBrilliance = None  # Float value
        self.obverseToning = None  # Float value
        self.reverseToning = None  # Float value
        self.revToningCovImg = None
        self.obToningCovImg = None
        self.obverseCoverage = None
        self.reverseCoverage = None
        self.brillianceHist = None
        self.obverseColors = []
        self.reverseColors = []

        # Unused for now
        self.mintLocation = None  # String
        self.mintYear = None  # Int

    # def coinInitialize(self, path, filenameObverse, filenameReverse):
    def coinInitialize(self, obverse, reverse):
        """
        Get the inputs for the machine learning prediction
        param: filename - the filename of the coin image. 1000x1000
        """
        try:
            self.obverseCoin.colorimg = obverse
            self.detailedResults.ogObverse = Image.fromarray(obverse)
            try:
                self.obverseCoin.reAdjustImage()
            except:
                self.obverseCoin.findcenter()
            self.obverseCoin.grayimg = cv2.cvtColor(self.obverseCoin.colorimg, cv2.COLOR_BGR2GRAY)
        except:
            self.obverseCoin.colorimg = None
            self.obverseCoin.grayimg = None

        try:
            self.reverseCoin.colorimg = reverse
            self.detailedResults.ogReverse = Image.fromarray(reverse)
            try:
                self.reverseCoin.reAdjustImage()
            except:
                self.reverseCoin.findcenter()
            self.reverseCoin.grayimg = cv2.cvtColor(self.reverseCoin.colorimg, cv2.COLOR_BGR2GRAY)
            self.reverseCoin.findcenter()
        except:
            self.reverseCoin.colorimg = None
            self.reverseCoin.grayimg = None

    def getConditionScore(self):
        """
        Get the condition score of the coin
        param: coin - the coin to get the condition score of, type CoinImage
        """
        self.obverseFeatures = []
        self.reverseFeatures = []
        obv_flat = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/CustomMasks/') + '\\' + 'obv_flat.jpg'
        rev_flat = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/CustomMasks/') + '\\' + 'rev_flat.jpg'
        obv_redorange = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/CustomMasks/') + '\\' + 'obv_red_orange_mask.jpg'
        rev_redorange = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/CustomMasks/') + '\\' + 'rev_red_orange_mask.jpg'
        obv_yellow = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/CustomMasks/') + '\\' + 'obv_yellow_mask.jpg'
        rev_yellow = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/CustomMasks/') + '\\' + 'rev_yellow_mask.jpg'
        obv_green = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/CustomMasks/') + '\\' + 'obv_green_mask.jpg'
        rev_green = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/CustomMasks/') + '\\' + 'rev_green_mask.jpg'

        # do the imperfection analysis with the flat mask
        flatObv = process_imperfection_image(coin=self.obverseCoin, mask=obv_flat)
        self.obverseFeatures.append(flatObv[1])
        self.detailedResults.flatObverse = flatObv[0]
        flatRev = process_imperfection_image(coin=self.reverseCoin, mask=rev_flat)
        self.reverseFeatures.append(flatRev[1])
        self.detailedResults.flatReverse = flatRev[0]

        # do the rest of the custom masks for feature evaluation
        obv_masks = [obv_redorange, obv_yellow, obv_green]
        obv_keys = ["highSigObverse", "lowSigObverse", "rimObverse"]
        rev_masks = [rev_redorange, rev_yellow, rev_green]
        rev_keys = ["highSigReverse", "lowSigReverse", "rimReverse"]
        for m in range(len(obv_masks)):
            result = process_one_image_custom_mask(coinimg=self.obverseCoin, mask=obv_masks[m])
            self.obverseFeatures.append(result[1])
            self.detailedResults.condMasks[obv_keys[m]] = result[0]
        for m in range(len(rev_masks)):
            result = process_one_image_custom_mask(coinimg=self.reverseCoin, mask=rev_masks[m])
            self.reverseFeatures.append(result[1])
            self.detailedResults.condMasks[rev_keys[m]] = result[0]

        self.detailedResults.conditionObverse = getConditionImage(self.obverseCoin)
        self.detailedResults.conditionReverse = getConditionImage(self.reverseCoin)

    def getToningScore(self):
        """
        Get the Toning score of the coin
        param: coin - the coin to get the toning score of, type CoinImage
        """
        self.obverseToning, self.obverseCoverage, self.obverseColors = getToningScore(coin=self.obverseCoin)
        self.reverseToning, self.reverseCoverage, self.reverseColors = getToningScore(coin=self.reverseCoin)
        self.obToningCovImg = getToningCoverageImage(self.obverseCoin, calcToningCoverage(eq.Calculate_HSV(self.obverseCoin.colorimg), coin=self.obverseCoin)[1])
        self.revToningCovImg = getToningCoverageImage(self.reverseCoin, calcToningCoverage(eq.Calculate_HSV(self.reverseCoin.colorimg), coin=self.reverseCoin)[1])

        self.detailedResults.toningScore = (self.obverseToning + self.reverseToning)/2
        self.detailedResults.obToningCovImgDR = self.obToningCovImg
        self.detailedResults.reToningCovImgDR = self.revToningCovImg
    def getColorScore(self):
        """
        Get the Color score of the coin
        param: coin - the coin to get the Color score of, type CoinImage
        """
        self.obverseBrilliance = getBrilliance_And_Percent_Silver(coin=self.obverseCoin)[0]
        self.reverseBrilliance = getBrilliance_And_Percent_Silver(coin=self.reverseCoin)[0]

        self.brillianceHist = getBrillianceHist()

        self.detailedResults.histBrillliance = self.brillianceHist
        self.detailedResults.brillianceScore = (self.obverseBrilliance + self.reverseBrilliance)/2

    def predictGrade(self):

        self.grader.PreProcessing(np.array(['EdgeFreq Flat Obverse',
                                            'EdgeFreq Flat Reverse',
                                            'EdgeFreq RedOrange Obverse',
                                            'EdgeFreq RedOrange Reverse',
                                            'EdgeFreq Yellow Obverse',
                                            'EdgeFreq Yellow Reverse',
                                            'EdgeFreq Green Obverse',
                                            'EdgeFreq Green Reverse',
                                            'Brilliance Obverse',
                                            'Brilliance Reverse',
                                            'Toning Obverse',
                                            'Toning Reverse'
                                            ]))
        self.grader.LoadModel()
        self.predictedGrade = self.grader.PredictGrade(np.array([self.obverseFeatures[0],
                                                                 self.reverseFeatures[0],
                                                                 self.obverseFeatures[1],
                                                                 self.reverseFeatures[1],
                                                                 self.obverseFeatures[2],
                                                                 self.reverseFeatures[2],
                                                                 self.obverseFeatures[3],
                                                                 self.reverseFeatures[3],
                                                                 self.obverseBrilliance,
                                                                 self.reverseBrilliance,
                                                                 self.obverseToning,
                                                                 self.reverseToning
                                                                 ]).reshape(1, -1))  # reshaped when doing the testpdf, if it gives an error on the website just get rid of the reshape
        self.detailedResults.conditionScore = self.predictedGrade
        print(self.predictedGrade)

    def generateDetailedResults(self):
        generateTemplate(self.detailedResults)

def runMSDCode(oImg, rImg):
    c = inputCoin()
    oImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2RGB)
    rImg = cv2.cvtColor(rImg, cv2.COLOR_BGR2RGB)

    c.coinInitialize(oImg, rImg)
    c.getConditionScore()
    c.getToningScore()
    c.getColorScore()
    c.predictGrade()
    c.generateDetailedResults()

if __name__ == '__main__':

    # TEST PDF
    c = inputCoin()
    oPath = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/images') + '\\'
    oImg = cv2.imread(oPath + "MSD_Proc_ob.jpg")
    oImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2RGB)
    rPath = os.path.abspath('MorganSilverDollar/Morgan_Dollar_main/images') + '\\'
    rImg = cv2.imread(rPath + "MSD_Proc_rev.jpg")
    rImg = cv2.cvtColor(rImg, cv2.COLOR_BGR2RGB)

    c.coinInitialize(oImg, rImg)
    c.getConditionScore()
    c.getToningScore()
    c.getColorScore()
    c.predictGrade()
    c.generateDetailedResults()

