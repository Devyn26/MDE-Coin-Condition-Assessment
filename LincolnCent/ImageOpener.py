import cv2
import os

def loadImages(mode, imageFolderPath):
    # Replace ./ with path to the cloned GitHub Repository. 
    # For example, in my case the full path is C:/Users/moham/OneDrive/Desktop/Senior/CoinCherrypicker/Images/"
    #imageFolderPath = "C:/Users/Mohammed/Desktop/SeniorDesign/CoinCherrypicker/CoinCherrypicker/Images/HSV/RB/"
    #imageFolderPath = "C:/Users/moham/OneDrive/Desktop/Senior/CoinCherrypicker/Images/HSV/ProtoCoins/"
    #Find the number of files (images) in image folder
    path, dirs, files = next(os.walk(imageFolderPath))
    file_count = len(files)
    print("Images Found: ", file_count)

    images = []

    # For every single image in the folder, append image name to imageFolderPath
    # Then, read and open the image.
    for i in files:

        imagePath = imageFolderPath + i
        if mode == 'color':
            images.append(cv2.imread(imagePath, cv2.IMREAD_COLOR))
        elif mode == 'grey':
            images.append(cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE))

        if __name__ == '__main__':
            windowName = "from David Lawrence Rare Coins " + i
            cv2.imshow(windowName, images[-1])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return images

if __name__ == '__main__':
    loadImages('color')
