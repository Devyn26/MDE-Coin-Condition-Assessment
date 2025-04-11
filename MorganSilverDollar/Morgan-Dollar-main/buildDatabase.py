"""
buildDatabase.py

Comprised of small helper functions used to parse data evaluated from each coin into a csv database

Author: Jasper Emick, Zymmorrah Myers
Date: 10 Mar 23
"""

import cv2
import pandas as pd
import numpy
import openpyxl
import csv
from tempfile import NamedTemporaryFile
import shutil


def parse_coin(imgName, descList):
    coinInfo = imgName.split(' ')

    print(coinInfo)
    if coinInfo[1][-3:] == '-CC':
        # Get only the year
        descList[0] = coinInfo[1][:-3]
        # -CC means Carson City
        descList[5] = "Carson City"
    else:
        descList[0] = coinInfo[1]
        descList[5] = "Not implemented"

    # Grader
    descList[1] = coinInfo[2]
    # Grade Type
    gType = "".join([c for c in coinInfo[3] if c.isalpha()])
    descList[2] = gType
    # Grade
    gradeNum = "".join([c for c in coinInfo[3] if c.isnumeric()])
    descList[3] = gradeNum
    # Inventory #
    descList[4] = coinInfo[-2]
    return descList


def compileScrapeData(coinImg, mint, toning):
    descriptionList = [None] * 17
    organizedList = parse_coin(coinImg, descriptionList)
    toning = toning[0]
    organizedList[5], organizedList[6] = mint, toning

    print(organizedList)
    df = pd.DataFrame([organizedList], columns=['Year',
                                                'Grader',
                                                'Grade Type',
                                                'Grade',
                                                'Inventory #',
                                                'Mint Location',
                                                'DL Toning Degree',
                                                'Brilliance Obverse',
                                                'Brilliance Reverse',
                                                'Toning Obverse',
                                                'Toning Reverse',
                                                'EdgeFreq Flat Obverse',
                                                'EdgeFreq Flat Reverse',
                                                'EdgeFreq Descriptive Obverse',
                                                'EdgeFreq Descriptive Reverse',
                                                'EdgeFreq RedOrange Obverse',
                                                'EdgeFreq RedOrange Reverse',
                                                'EdgeFreq Yellow Obverse',
                                                'EdgeFreq Yellow Reverse',
                                                'EdgeFreq Green Obverse',
                                                'EdgeFreq Green Reverse'])

    temp = pd.read_csv("image_database.csv", usecols=['Inventory #'])
    v = temp.isin([int(organizedList[4])]).any()
    # Prevent duplicates, limited by inventory number
    if not v.bool():
        # Set to append to the bottom row of the dataset
        df.to_csv("image_database.csv", mode='a', index=False, header=False)


def compileHSVData(avgHSV, percent, classification, img):
    coinInfo = img.split(' ')
    if coinInfo.count('') > 0:
        coinInfo.remove('')

    face = coinInfo[-1][:-4]
    inventory = coinInfo[-2]

    csvFile = "image_database.csv"
    database = pd.read_csv(csvFile)

    # print(inventory)
    d = database.loc[database['Inventory #'].isin([int(inventory)])]
    print(d.index)
    database.at[d.index[0], '% Silver'] = percent[1]

    database.to_csv("image_database.csv", index=False)


def compileConditionData(density, filename, quadrant, colname):
    coinInfo = filename.split(' ')
    if coinInfo.count('') > 0:
        coinInfo.remove('')

    face = coinInfo[-1][:-4]
    inventory = coinInfo[-2]

    csvFile = "image_database.csv"
    database = pd.read_csv(csvFile)
    #print(database)
    # print(inventory)
    d = database.loc[database['Inventory #'].isin([int(inventory)])]
    if not d.empty:
        if face == "obverse":
            if quadrant is True:
                database.at[d.index[0], 'TR Obverse'] = density[0]
                database.at[d.index[0], 'BR Obverse'] = density[1]
                database.at[d.index[0], 'BL Obverse'] = density[2]
                database.at[d.index[0], 'TL Obverse'] = density[3]
            else:
                database.at[d.index[0], colname + ' Obverse'] = density

        elif face == "reverse":
            if quadrant is True:
                database.at[d.index[0], 'TR Reverse'] = density[0]
                database.at[d.index[0], 'BR Reverse'] = density[1]
                database.at[d.index[0], 'BL Reverse'] = density[2]
                database.at[d.index[0], 'TL Reverse'] = density[3]
            else:
                database.at[d.index[0], colname + ' Reverse'] = density
        else:
            print("Not working stupid")
            return

        database.to_csv(csvFile, index=False)

def compileToningData(score, filename):
    coinInfo = filename.split(' ')
    if coinInfo.count('') > 0:
        coinInfo.remove('')

    face = coinInfo[-1][:-4]
    inventory = coinInfo[-2]

    csvFile = "image_database.csv"
    database = pd.read_csv(csvFile)

    # print(inventory)
    temp = pd.read_csv("image_database.csv", usecols=['Inventory #'])
    v = temp.isin([int(inventory)]).any()
    d = database.loc[database['Inventory #'].isin([int(inventory)])]
    if not d.empty:
        if face == "obverse":
            database.at[d.index[0], 'Brilliance Obverse'] = score
        else:
            database.at[d.index[0], 'Brilliance Reverse'] = score
        database.to_csv("image_database.csv", index=False)

def check(score, filename):
    coinInfo = filename.split(' ')
    if coinInfo.count('') > 0:
        coinInfo.remove('')

    face = coinInfo[-1][:-4]
    inventory = coinInfo[-2]

    csvFile = "image_database.csv"
    database = pd.read_csv(csvFile)

    # print(inventory)
    d = database.loc[database['Inventory #'].isin([int(inventory)])]
    if (database.at[d.index[0], 'Toning Reverse'] > -1) and (database.at[d.index[0], 'Toning Reverse'] > -1):
        print("Not Empty")
        return False
    else:
        print("Empty")
        return True