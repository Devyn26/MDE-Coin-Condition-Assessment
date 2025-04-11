import cv2 as cv2
import os
import numpy as np
from ToningProcessing import getToningScore
from Brilliance import getBrilliance_And_Percent_Silver
from buildDatabase import compileToningData
from buildDatabase import check
from CoinImage import CoinImage


def getAllToningData():

    dirname = os.path.abspath('ScrapedImagesHigh/ScrapedImagesHigh/reverse') + '\\'

    fileList = os.listdir(dirname)

    print(fileList)
    print("For my next trick I will process " + str(len(fileList)) + " Morgan Dollar images")
    #TS = 0
    Brilliance = 0
    Empty = True
    for index, filename in enumerate(fileList):
        #if(index >= 3359):
            print(index, "----------------------------------------------------------------")
            print(dirname + filename)
            coin = CoinImage()
            coin.load(dirname + filename)
            coin.filename = filename
            #TS = getToningScore(coin)
            Brilliance = getBrilliance_And_Percent_Silver(coin)[0]
            print("This is Brilliance Printed: ", Brilliance)
            # Writes evaluated condition data to csv database
            compileToningData(score=Brilliance, filename=filename)



if __name__ == '__main__':
    getAllToningData()
