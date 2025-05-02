"""
ToningProcessing.py

Purpose: This file calculates the toning grade

Explanation of code: First the image of the coin goes through the toning color palet (A range of colors
that are commonly found in Morgan Silver Dollars) to find what colors are in the coin. If the coin is
toned some similar colors get counted, so to prevent that we grouped certain colors together and that
way they would only get counted once. Next, we receive a percentage of coverage from ToningCoverage.py
and based on how much the coin is covered that will receive are score of 1-5 and has a %65 effect on the
overall toning score. In addition, we used the colors found previously and based on how many are in the
coin that will receive a score of 1-5 and has a %35 effect on the overall toning score.

How To Run File: Have an arbitrary value equal getToningScore with an image path of the users choice plugged
in. Then, print that arbitrary value, and the user will be able to see what toning score the image of their
choice gives.

Author: Zymmorrah Myers and Matthew Donlon
Date: 12 Mar 2023
"""
import sys
from .Brilliance import getBrilliance_And_Percent_Silver
from .ToningCoverage import getToningCoverage
from .colorPalette import find_colors
from .CoinImage import CoinImage
import os

def getToningScore(coin):

    coverage = getToningCoverage(coin)[0]
    print(coverage)
    color_list = []
    brilliance_score = 0
    if coverage >= 10:
        colors_list = find_colors(coin)
        white = 0
        brown = 0
        purple = 0
        blue = 0
        green = 0
        orange = 0
        yellow = 0
        red = 0
        black = 0
        pink = 0
        for i in color_list:
            print(i)
            if i in ['Light Gold', 'Medium Gold', 'Lemon Yellow']:
                white = 1
            elif i in ['Amber', 'Russet']:
                brown = 1
            elif i in ['Burgundy', 'Magenta Blue', 'Deep Magenta', 'Deep Purple']:
                purple = 1
            elif i in ['Cobalt Blue', 'Light Cyan Blue', 'Blue', 'Deep Blue']:
                blue = 1
            elif i in ['Pale Mint Green', 'Blue Green', 'Emerald Green', 'Deep Green']:
                green = 1
            elif i in ['Sunset Yellow', 'Gold']:
                yellow = 1
            elif i == 'Orange':
                orange = 1
            elif i in ['Red', 'Dark Red', 'Wine Red']:
                red = 1
            elif i in ['Glossy Black', 'Dull Black']:
                black = 1
            elif i in ['Magenta', 'MediumMagenta']:
                pink = 1
        num_colors = white + brown + purple + blue + green + yellow + orange + red + black + pink
        #print("Coverage: ", coverage)
        #print("Num Colors: ", num_colors)
    else:
        brilliance_score = getBrilliance_And_Percent_Silver(coin)[0]
        #print("Coverage: ", coverage)
        #print("Brilliance: ", brilliance_score)
        return 0, 0, ["Silver"]

    toning_score_cvg = 0

    if coverage > 65:
        toning_score_cvg = 5
    elif coverage > 55:
        toning_score_cvg = 4
    elif coverage > 45:
        toning_score_cvg = 3
    elif coverage > 35:
        toning_score_cvg = 2
    elif coverage > 25:
        toning_score_cvg = 1

    toning_score_clr = 0

    if num_colors >= 6:
        toning_score_clr = 5
    elif num_colors >= 5:
        toning_score_clr = 4
    elif num_colors >= 4:
        toning_score_clr = 3
    elif num_colors >= 3:
        toning_score_clr = 2
    elif num_colors >= 2:
        toning_score_clr = 1

    toning_score = round(((toning_score_cvg * 2) * 0.65 + (toning_score_clr * 2) * 0.35) / 2, 1)
    # print(toning_score)

    return toning_score, coverage, colors_list


if __name__ == '__main__':
    image = os.path.abspath('ScrapedImages/obverse/Morgan 1878 ICG MS64 2275098 obverse.jpg')
    print(image)
    coin = CoinImage()
    coin.load(image)
    getToningScore(coin=coin)
