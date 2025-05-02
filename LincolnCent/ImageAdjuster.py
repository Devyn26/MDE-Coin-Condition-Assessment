'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
'''

from .ImageOpener import loadImages
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def NameOfFile():
    imageFolderPath = "D:/4805/code/CoinCherrypicker/Images/"
    path, dirs, files = next(os.walk(imageFolderPath))
    lst=[]
    for i in files:
        lst.append(i[:-4])
    return lst

def Histogram_drawing(HSL_Value):
    plt.subplot(1, 3, 1)
    plt.hist(HSL_Value[0], bins=80, range=(0, 40), histtype='step', color='r', label='Hue')
    plt.title("Hue")
    plt.xlabel("Degree 0-360")
    plt.ylabel("Frequency")
    plt.legend()

def canny(img):
    img = cv2.GaussianBlur(img,(5,5),sigmaX=1.5,sigmaY=1.5)
    edge = cv2.Canny(img, 100, 200)

    return edge

def sobel(img):
    edge = cv2.convertScaleAbs(cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3))

    return edge

def gaussian(img):
    img = cv2.GaussianBlur(img,(0,0),sigmaX=1.375,sigmaY=1.375)

    return img

def sobelOfGaussian(img):
    img = cv2.GaussianBlur(img,(0,0),sigmaX=1.375,sigmaY=1.375)
    edge = cv2.convertScaleAbs(cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3))

    return edge

def binarize(img, mode):
    img = cv2.GaussianBlur(img,(5,5),sigmaX=1.5,sigmaY=1.5)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,5,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,5,2)
    # titles = ['Original Image', 'Global Thresholding (v = 127)',
    #             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    # images = [img, th1, th2, th3]
    # for i in range(4):
    #     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])
    # plt.show()

    if mode == 'mean':
        return th2
    elif mode == 'gaussian':
        return th3
    else:
        return np.ERR_DEFAULT

def laplacianOfGaussian(img):
    # Apply Gaussian Blur
    img = cv2.GaussianBlur(img,(0,0),sigmaX=1.375,sigmaY=1.375)
    
    # Apply Laplacian operator in some higher datatype
    laplacian = cv2.Laplacian(img, ksize=3, ddepth=cv2.CV_16S)
    filtered_image = cv2.convertScaleAbs(laplacian)

    return filtered_image

def Image_HSV():
    img = loadImages('color')
    NameFile= NameOfFile()
    i = 0
    list_sat=[]
    for Image in img:
         # cropped=Image[300:400,300:500]  #this is for RGB
         print(NameFile[i])
         croppedForGray = Image[450:600, 700:850]  # y: y+h, x:x+w this is above the data
         gray = cv2.cvtColor(croppedForGray, cv2.COLOR_BGR2GRAY)
         # INPAINT
         mask1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
         cropped = cv2.inpaint(croppedForGray, mask1, 0.1, cv2.INPAINT_TELEA)


         # cropped = Image[450:600,700:850]  # y: y+h, x:x+w this is above the data


                    #    y1  y2  x1   x2
         # above the data
         cv2.rectangle(Image, (700, 450), (850, 600), (0, 0, 250), 3)   # x left y top  x right  y bottom
                              # x1   y1    x2   y2

         #above the Liberty
         # cv2.rectangle(Image, (130, 330), (280, 480), (0, 0, 250), 3)   # x left y top  x right  y bottom
                              # x1   y1    x2   y2

         # cv2.imshow('image',Image)
         # plt.show()
         # cv2.waitKey(0)
         # cv2.destroyAllWindows()

         # HSL_Value=Calculate_HSV(cropped)

         # list_sat.append(GetSaturationQuestion(HSL_Value))

         # Questionable(HSL_Value)
         # Histogram_drawing(HSL_Value)


         # mean_median_value(HSL_Value)      #print mean median
         i+=1

    x = []
    i = 0
    for item in list_sat:
        x.append(i)
        i += 1
    plt.scatter(x, list_sat)
    plt.title("Mean Saturation")
    plt.xlabel("number")
    plt.ylabel("Mean Saturation Value")
    plt.show()
    # print("Hue:{0} saturation:{1} L:{2}".format(H_hsv_degree,Saturation,L))

    return flatendImg

if __name__ == '__main__':
    img = loadImages('grey', './Git-REpo-Indian-Head/Images/Obverse/')
    for i in img:
        flat = sobel(i)
        (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 15))
        ax1.title.set_text('Original Image')
        ax1.imshow(i, cmap='gray')
        ax2.title.set_text('Laplacian Filtered Image')
        ax2.imshow(flat, cmap='gray')
        plt.show()
