import requests
from bs4 import BeautifulSoup
import cv2
import urllib
import shutil

#import ImageHSV

from PIL import Image
import time
import os
import ssl

""" 
FOR MADALYN:
❤︎  https://github.com/mohamhumadi/CoinCherrypicker/tree/Madalyns-Branch

FOR TEAM NOBLE RATS:
❤︎  to run this code you must use the command "python3 ./blahblah" in the terminal, i dunno why so dont ask
❤︎  the only line you need to edit is line directly below this comment block

PATCH NOTES (updated as of 12/3/2021):

❤︎  this image scraper takes both the obverse and reverse images of pennies from the following url:
    https://www.davidlawrence.com/en-us/us-coins/united-states-small-cents?tb=653000&PriceLow=0&PriceHigh=100000000&GradeLow=0&GradeHigh=70&SeriesName=lincoln-cents,lincoln-cents-(proof),lincoln-cents-memorial-reverse,lincoln-cents-memorial-reverse-proof,lincoln-cents-bicentennial-shield-reverse,lincoln-cents-bicentennial-shield-reverse-proof,lincoln-cents-special-strikes-special-mint-sets&NuTilt=all&Sort=default&Status=all&ResultsPerPage=163

❤︎  the image scraper saves the images into the SCRAPED_IMAGES_FILE_PATH directories defined below:
❤︎  "OBVERSE": contains all of the obverse images of the pennies
❤︎  "REVERSE": contains all of the reverse images of the pennies
❤︎  "OBVERSE AND REVERSE": contains both the obverse images and reverse images of the pennies

❤︎  image scraper ignores any rogue images that are not of coins (example: "Proof" images)

❤︎  potentially scrape other coins like indian head??????

❤︎  seperate 1909 coins

❤︎  as of 12/3/2021, this scraper does NOT ignore the STEEL coins in its scraping, functionality in process of being implemented (using HSV)

"""

"""
PATCH NOTES (Last updated 10/11/2022)
New version for 2022-2023 Indian Head Cent Team

Target Directory "MC" changed to "Other", will contain Copper-Nickel and unassigned coins

Updated the output file nomenclature
    <InventoryNumber Grader YearMintmark Grade Color EyeAppeal Toning PictureType (Special Notes)>.jpg
    Ex. "2348850 PCGS 1860 MS64+ CN 5 2 Reverse (Pointed Bust).jpg"

Original comments retained for archival purposes

"""


# FOR CURRENT USER: Change the paths here to correspond to your desired file system setup
SCRAPED_IMAGES_FILE_PATH = "C:\\Users\\Adam\\Desktop\\ECE 4805\\Code\\MDE_IHC_Scraper\\Coins"
SCRAPED_IMAGES_FILE_PATH_RD = "C:\\Users\\Adam\\Desktop\\ECE 4805\\Code\\MDE_IHC_Scraper\\Coins\\RD"
SCRAPED_IMAGES_FILE_PATH_BN = "C:\\Users\\Adam\\Desktop\\ECE 4805\\Code\\MDE_IHC_Scraper\\Coins\\BN"
SCRAPED_IMAGES_FILE_PATH_RB = "C:\\Users\\Adam\\Desktop\\ECE 4805\\Code\\MDE_IHC_Scraper\\Coins\\RB"
SCRAPED_IMAGES_FILE_PATH_OTHER = "C:\\Users\\Adam\\Desktop\\ECE 4805\\Code\\MDE_IHC_Scraper\\Coins\\CN"




# ------------------------------------------------------ DO NOT TOUCHY ------------------------------------------------------
def getdata(url):
    r = requests.get(url)
    return r.text

def getMetadata(url):
    
    ssl._create_default_https_context = ssl._create_unverified_context

    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    layer1 = soup.body.find("div", {"id":"page-wrapper"}).find("section", {"class":"content blog"})
    layer2 = layer1.div.div.div.find("div", {"class":"col-xs-12 col-sm-6 col-md-8"})
    
    InvNum = layer2.h4.div.find_all("strong")[2].contents[1]
    
    # IHC: This one was weird, counts the instances of filled star objects to determine Eye Appeal rating
    EyeAppeal = len(layer2.find_all("i", {"class": "fa fa-star"}))
    
    Toning = layer2.h4.div.find_all("strong")[1].contents[1]
    
    # IHC: remove extraneous whitespace and the last three characters from Toning (e.g. "2/10" -> "2")
    Toning = Toning.strip()
    Toning = Toning[:-3]

    #print(InvNum)
    #print(EyeAppeal)
    #print(Toning)

    return InvNum, EyeAppeal, Toning

def getBackOfCoin(url, image_title):

    ssl._create_default_https_context = ssl._create_unverified_context

    ignored = "?MaxWidth=250&MaxHeight=250&Mode=Pad&Scale=UpscaleCanvas"
    # takes in the elements data from the new site
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    layer1 = soup.body.find("div", {"id":"page-wrapper"}).find("section", {"class":"content blog"})
    layer2 = layer1.div.div.div.find("div", {"class":"col-xs-12 col-sm-6 col-md-4"})
    reverse_url = layer2.find("div", {"class": "slider slider-nav"}).find_all("div")[1].img['src']
    
    reverse_url = reverse_url[:-len(ignored)]
    image_title += " Reverse"
    image_title += ".jpg"
    image_title = image_title.replace('/', '-')
    image_title = image_title.replace(':', ';')

    proof = "Proof"
    if proof not in image_title:
        try:
            urllib.request.urlretrieve(reverse_url, image_title) # saving the image as the image title
            moveToScrapedImagesDir(image_title)
        except:
            print(reverse_url)
            print(image_title)
            exit()

    return 0

def moveToScrapedImagesDir(file_name):
    # get the name of the current directory
    CURRENT_FILE_PATH = os.getcwd()
    image_path = CURRENT_FILE_PATH + "\\" + file_name

    if "RD" in file_name:
        spliced_path = SCRAPED_IMAGES_FILE_PATH_RD + file_name    
        shutil.move(image_path, spliced_path) 
    elif "BN" in file_name:
        spliced_path = SCRAPED_IMAGES_FILE_PATH_BN + file_name    
        shutil.move(image_path, spliced_path) 
    elif "RB" in file_name:
        spliced_path = SCRAPED_IMAGES_FILE_PATH_RB + file_name    
        shutil.move(image_path, spliced_path) 
    else:
        spliced_path = SCRAPED_IMAGES_FILE_PATH_OTHER + file_name    
        shutil.move(image_path, spliced_path) 

def moveImagesToReverseOrObverseFiles():

    # find the number of files (images) in image folder
    path, dirs, files = next(os.walk(SCRAPED_IMAGES_FILE_PATH))
    file_count = len(files)
    print("Images Found: ", file_count)

    obverse_images = []
    reverse_images = []
#got
    # add the respective images to lists depending on obverse or reverse
    for i in files:
        imagePath = SCRAPED_IMAGES_FILE_PATH + i
        if "Obverse" in imagePath:
            obverse_images.append(imagePath)
        elif "Reverse" in imagePath:
            reverse_images.append(imagePath)

    # checks to see if the image scraper has already been ran
    obverse_path = SCRAPED_IMAGES_FILE_PATH + "OBVERSE"
    reverse_path = SCRAPED_IMAGES_FILE_PATH + "REVERSE"
    both_path = SCRAPED_IMAGES_FILE_PATH + "OBVERSE AND REVERSE"

    is_obverse_file = os.path.isdir(obverse_path)
    is_reverse_file = os.path.isdir(reverse_path)
    is_both_file = os.path.isdir(both_path)

    if is_obverse_file == False:
        os.mkdir(obverse_path)
    
    if is_reverse_file == False:
        os.mkdir(reverse_path)

    if is_both_file == False:
        os.mkdir(both_path)

    # puts the obverse and reverse images in the correct folders
    for obverse_image in obverse_images:
        obverse_image_new = obverse_image[len(SCRAPED_IMAGES_FILE_PATH):]
        obverse_path_new = obverse_path + "/" + obverse_image_new
        shutil.copy(obverse_image, obverse_path_new) 

    for reverse_image in reverse_images:
        reverse_image_new = reverse_image[len(SCRAPED_IMAGES_FILE_PATH):]
        reverse_path_new = reverse_path + "/" + reverse_image_new
        shutil.copy(reverse_image, reverse_path_new) 

    for i in files:
        temp_path = SCRAPED_IMAGES_FILE_PATH + i
        both_path_new = both_path  + "/" + i
        shutil.move(temp_path, both_path_new) 

def main():
    # The ignored variable is whatis ignored from the image link. This results in the full, high-res image pulled from the DLRC website.
    imageCount = 0
    caseImage = False
    ignored = "?MaxWidth=250&MaxHeight=250&Mode=Pad&Scale=UpscaleCanvas"
    htmldata = getdata("https://www.davidlawrence.com/en-us/us-coins/united-states-small-cents?tb=165727&PriceLow=0&PriceHigh=100000000&GradeLow=0&GradeHigh=70&SeriesName=indian-cents&NuTilt=all&Sort=default&Status=BuyItNow&ResultsPerPage=120&page=2")
    soup = BeautifulSoup(htmldata, 'html.parser')

    # donmt fucking delete this line its ksut foprf madalyns sake but i need it ep[lase dear god]
    ssl._create_default_https_context = ssl._create_unverified_context

    coin_wrappers = soup.find_all("div", {"class": "col-xs-6 col-sm-4"})
    
    print(len(coin_wrappers))

    for coin_wrapper in coin_wrappers:      
        url = coin_wrapper.article.find_all("img", {"class": "img-responsive sSlab disabled"})[0]['src']
        image_title = coin_wrapper.article.find_all("a", {"class": "product-permalink"})[1].string.strip()
        
        current_images = os.listdir()
        counter = 1
        # print(current_images)
        if image_title + '.jpg' in current_images:
            while image_title + '-' + str(counter) + '.jpg' in current_images:
                counter += 1
            image_title += '-' + str(counter)
            #print(image_title)
        
        base_url = "https://www.davidlawrence.com"
        unique_url = base_url + coin_wrapper.find("a")['href']

    # IHC: ----InventoryNum, Eye Appeal, Toning scores----

        InvNum, EyeAppeal, Toning = getMetadata(unique_url)
        image_title = str(InvNum) + " " + image_title + " " + str(EyeAppeal) + " " + Toning

        # -------------- REVERSE --------------

        
        getBackOfCoin(unique_url, image_title)

        # -------------- OBVERSE --------------

        url = url[:-len(ignored)]
        image_title += " Obverse"
        image_title += ".jpg"
        image_title = image_title.replace('/', '-')
        image_title = image_title.replace(':', ';')

        # -------------- SAVE IMAGE --------------
        proof = "Proof"
        if proof not in image_title:
            try:
                urllib.request.urlretrieve(url, image_title)
                moveToScrapedImagesDir(image_title)
            except:
                print(url)
                print(image_title)
                exit()

        time.sleep(.5)

        """# -------------- IGNORE SILVER COINS --------------
        spliced_name = SCRAPED_IMAGES_FILE_PATH + image_title
        img = cv2.imread(spliced_name,0)
        #image = Image.open(spliced_name)
        #image.show()

        cropped = img[450:600,700:850]
        HSV_val = ImageHSV.Calculate_HSV(cropped)
        print(HSV_val)
        """        
    
    #moveImagesToReverseOrObverseFiles()


if __name__ == "__main__":
    main()