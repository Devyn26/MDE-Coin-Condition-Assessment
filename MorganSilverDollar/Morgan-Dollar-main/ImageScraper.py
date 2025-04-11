"""
ImageScraper.py

Runs through every available Morgan Silver dollar listed on the David Lawrence Catalog; scrapes each image and relevant
coin data to be used in analysis and machine learning.

Original Author: MDE 2021-2022 Coin CherryPicker Team
Date: 3 Dec 2021
Modified By: Jasper Emick
Date: 10 Mar 23
"""

import requests
from bs4 import BeautifulSoup
import time
import os
import ssl
from buildDatabase import compileScrapeData


def getdata(url):
    r = requests.get(url)
    return r.text


def removeGarbage(image_title):
    """
    Tries to remove all irrelevant characters/strings from the title of the Coin given by David Lawrence.  Because of
    the unpredictability for this, some silly things will still slip through sometimes.
    """
    index = image_title.find('(')

    if index != -1:
        image_title = image_title[:index - 1]

    image_title = image_title.replace('.', '')
    image_title = image_title.replace(':', '')
    image_title = image_title.replace('$1 ', '')
    image_title = image_title.replace('/', '-')
    image_title = image_title.replace('7-8TF ', '')
    image_title = image_title.replace('7TF ', '')
    image_title = image_title.replace('8TF ', '')
    image_title = image_title.replace('8-7  ', '')
    image_title = image_title.replace('TF ', '')
    image_title = image_title.replace(' DMPL', '')
    image_title = image_title.replace(' Details', '')
    image_title = image_title.replace(' PL', '')
    image_title = image_title.replace(' Fine', '')
    image_title = image_title.replace(' Hot Lips', '')
    image_title = image_title.replace('Morgan ', '')
    image_title = image_title.replace('Hot 50 ', '')
    image_title = image_title.replace(' *Star*', '')
    image_title = image_title.replace('Roll of ', '')
    image_title = image_title.replace(' Strong', '')
    #image_title = image_title.replace('Reverse of ', '')

    exString = ""
    start = None
    for i, c in enumerate(image_title):
        if c == 'e' and i != image_title[len(image_title) - 1] and start is None:
            if image_title[i + 1] == 'x':
                start = i

    if start is not None:
        exString = image_title[start:]
    image_title = image_title.replace(exString, '')

    return image_title


def moveToScrapedImagesDir(responseURL, file_name, face):
    if not responseURL.ok:
        print("Failed to get url, perhaps the image is no longer in the database")
        return -1

    # Check that the folder exists
    if not os.path.exists("ScrapedImages"):
        os.makedirs("ScrapedImages")
        os.makedirs("ScrapedImages/obverse")
        os.makedirs("ScrapedImages/reverse")

    file_path = None
    if face == "o" and not os.path.isfile("ScrapedImages/obverse/" + file_name):
        file_path = os.path.join("ScrapedImages/obverse", file_name)
    elif face == "r" and not os.path.isfile("ScrapedImages/reverse/" + file_name):
        file_path = os.path.join("ScrapedImages/reverse", file_name)

    if file_path is not None:
        with open(file_path, 'wb') as file:
            for chunk in responseURL.iter_content(chunk_size=1024 * 8):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())
    else:
        print("Image is already in directory")


def main():
    # This is to remove the downscaling applied to the images so that all image sizes are consistent
    IGNORED = -56
    htmldata = getdata(
        "https://www.davidlawrence.com/en-us/us-coins/united-states-silver-dollars?tb=322810&PriceLow=0&PriceHigh"
        "=100000000&GradeLow=0&GradeHigh=70&SeriesName=morgan-dollars&NuTilt=all&Sort=default&Status=all"
        "&ResultsPerPage=1000")
    soup = BeautifulSoup(htmldata, 'html.parser')

    # I'm not sure what this does but someone on the previous team was very passionate about having this stay here
    ssl._create_default_https_context = ssl._create_unverified_context

    coin_wrappers = soup.find_all("div", {"class": "col-xs-6 col-sm-4"})

    # Loops for every relevant Morgan Dollar Coin in the David Lawrence catalog
    for coin_wrapper in coin_wrappers:
        # Gets the important details for the naming the coin
        image_title = coin_wrapper.article.find_all("a", {"class": "product-permalink"})[1].string.strip()
        """ 
        *** Adjusts the image title such that there are no conflicts when reading, a bit sloppy right now, 
        *** will clean up later.
        """
        image_title = str(image_title)
        if not image_title.find('Lot'):
            continue
        new_title = removeGarbage(image_title)
        print(new_title)

        # Gets the unique url for the coin, each coin is identified by an inventory number
        inventoryUrl = coin_wrapper.article.find_all('a', href=True)[0]['href']
        inventory = inventoryUrl[11:]
        base_url = "https://www.davidlawrence.com"
        """ 
        *** Accesses the specified coin page within the website catalog to obtain the urls of the observe and reverse
        *** sides of each coin.
        """
        newUrl = base_url + inventoryUrl
        internalData = getdata(newUrl)
        internalSoup = BeautifulSoup(internalData, 'html.parser')

        # Finds the image urls within the parsed URL data of the page
        observeUrl = internalSoup.find_all("img", {"class": "img-responsive"})[1]["src"][:IGNORED]
        reverseUrl = internalSoup.find_all("img", {"class": "img-responsive"})[2]["src"][:IGNORED]

        getMint = internalSoup.find_all("div", {"class": "col-xs-12 col-sm-12 col-md-6"})[1]
        degreeToning = internalSoup.find('span', text="Degree of Toning: ").find_next_sibling(text=True).strip()
        strMint = str(getMint)
        mintLoc = "N/A"
        for item in strMint.split("\n"):
            if "Mint Location" in item:
                mintLoc = item[32:-5]

        # Adjusts the titles for each side of the coin
        observe_title = "Morgan " + new_title + " " + str(inventory) + " obverse" + ".jpg"
        reverse_title = "Morgan " + new_title + " " + str(inventory) + " reverse" + ".jpg"

        compileScrapeData(observe_title, mintLoc, degreeToning)

        print(observeUrl)
        moveToScrapedImagesDir(requests.get(observeUrl), observe_title, "o")
        time.sleep(.1)

        print(reverseUrl)
        moveToScrapedImagesDir(requests.get(reverseUrl), reverse_title, "r")
        time.sleep(.1)


if __name__ == "__main__":
    main()
