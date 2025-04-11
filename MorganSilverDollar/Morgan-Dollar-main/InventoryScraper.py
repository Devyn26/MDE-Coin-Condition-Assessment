"""
InventoryScraper.py

Runs through every available inventory space on the David Lawrence Coin Catalog to search for unlisted Morgan Silver
Dollars; scrapes each image and relevant coin data to be used in analysis and machine learning.

Author: Jasper Emick
Date: 10 Mar 23
"""
import pandas as pd
from bs4 import BeautifulSoup
import requests
from ImageScraper import removeGarbage, moveToScrapedImagesDir
import time
from buildDatabase import compileScrapeData


def getdata(url):
    r = requests.get(url)
    return r.text


def processCoinData(start):
    df = pd.read_csv('morgan_urls.csv')

    urlList = df.to_numpy()

    for url in urlList[start:]:

        urlStr = url[0]

        urlData = getdata(urlStr)
        soup = BeautifulSoup(urlData, 'html.parser')
        # Gets the silly little coin title and cleans it up
        rawInspect = str(soup.find_all("h1")[0])
        titleList = rawInspect.split(">")
        rawTitle = titleList[2][:-8]
        print(rawTitle)
        if rawTitle.find("Lot") != -1:
            continue
        cleanedTitle = removeGarbage(rawTitle)
        print(cleanedTitle)

        degreeToning, inventory, mintLocation = "", "", ""

        # Obtains remaining relevant coin info
        stuffList = soup.find_all("strong")
        for stuff in stuffList:

            if stuff.text.find("Degree of Toning") != -1:
                degreeToning = stuff.text.split('\t')[-1]
            elif stuff.text.find("Inventory") != -1:
                inventory = stuff.text.split(': ')[-1]
            elif stuff.text.find("Mint Location") != -1:
                mintLocation = stuff.next_sibling.text

        # Gets the obverse and reverse images of each coin
        #print(soup.article.find_all("img", {"class": "img-responsive"}))
        sizeAdjust = "?MaxWidth=1000&MaxHeight=1000&Mode=Pad&Scale=UpscaleCanvas"
        obverseUrl = soup.article.find_all("img", {"class": "img-responsive"})[0]["src"] + sizeAdjust
        reverseUrl = soup.article.find_all("img", {"class": "img-responsive"})[1]["src"] + sizeAdjust

        observe_title = "Morgan " + cleanedTitle + " " + str(inventory) + " obverse" + ".jpg"
        reverse_title = "Morgan " + cleanedTitle + " " + str(inventory) + " reverse" + ".jpg"

        compileScrapeData(observe_title, mintLocation, degreeToning)

        print(obverseUrl)
        moveToScrapedImagesDir(requests.get(obverseUrl), observe_title, "o")
        time.sleep(.1)  # This is to reduce the strain put on the website, should probably be higher but I'm a demon

        print(reverseUrl)
        moveToScrapedImagesDir(requests.get(reverseUrl), reverse_title, "r")
        time.sleep(.1)


def validateURL():
    # No img ex: https://www.davidlawrence.com/rare-coin/2450000
    # different coin ex: https://www.davidlawrence.com/rare-coin/2460000
    # Invalid Img ex: https://www.davidlawrence.com/rare-coin/2450007
    # Literally Nothing ex: https://www.davidlawrence.com/rare-coin/2440324

    START_INVENTORY = 1927600
    MAX_INVENTORY = 2000000

    # ME - 2000000 - 2150000, 1850000 - 2000000
    # Lizzieh - 2150000 - 2300000 (And if you want to do more then just keep going until like 2400000! ! !!  !!)

    # Current checkpoint - 2130000 (Jasper) also 1890000

    morganUrls = []

    # Loops through the set range of inventory numbers
    for i in range(START_INVENTORY, MAX_INVENTORY):
        url = "https://www.davidlawrence.com/rare-coin/" + str(i)

        morganSilly = getdata(url)

        soup = BeautifulSoup(morganSilly, 'html.parser')
        # Check if the coin is a MSD
        coin_identifier = soup.find_all("span", {"class": "label label-success btn-white-text"})

        if len(coin_identifier) != 0:
            searchValue = str(coin_identifier[0]).find("Morgan Dollars")
            if searchValue != -1:
                # Check if proof, if true then skip
                if str(coin_identifier[0]).find("(Proof)") != -1:
                    continue

                # Parses inspection info to look for images on the page
                image_wrapper = soup.find_all("div", {"class": "col-xs-12 col-sm-6 col-md-4"})[0]
                elementsList = str(image_wrapper).split("<")

                # Checks that there are at least two images provided for the MSD
                imgCount = 0
                for e in elementsList:
                    if e.find("img-responsive") != -1:
                        imgCount += 0.5

                if imgCount >= 2:
                    print("Valid Morgan")
                    morganUrls.append(url)

        # Writes to csv for every 200 inventory jumps, this is in-case you want to take a break or if something explodes
        if i % 200 == 0 and i > 99:
            print("write " + str(i))
            # Creates and writes to an extremely bare bones csv file but WHO CARES!
            df = pd.DataFrame(morganUrls, columns=["Valid URLS"])
            df.to_csv('morgan_urls.csv', mode='a', index=False, header=False)
            # Empty List to prevent repeats
            morganUrls = []

        time.sleep(.1)


if __name__ == "__main__":
    validateURL()
    # validateURL()
