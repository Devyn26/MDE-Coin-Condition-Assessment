'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
'''

import cv2
import os
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
from GUI import *

# Find the Path of an Image
def findImagePath():
    # imageFolderPath = "C:/Users/moham/OneDrive/Desktop/Senior/CoinCherrypicker/Images/HSV/ProtoCoins/"
    imageFolderPath = obversePath
    return imageFolderPath

# Archaic code. Used to find all of the images in a folder in order to analyze them all at once. This is now done in ImageOpener
def NameOfFile():
    imageFolderPath = "C:/Users/moham/OneDrive/Desktop/Senior/CoinCherrypicker/Images/HSV/ProtoCoins/"
    path, dirs, files = next(os.walk(imageFolderPath))
    lst=[]
    for i in files:
        lst.append(i[:-4])
    return lst

# Display the HSV histogram for the image
def Histogram_drawing(HSV_Value):
    plt.subplot(1, 3, 1)
    plt.hist(HSV_Value[0], bins=80, range=(0, 360), histtype='stepfilled', color='r', label='Hue')
    plt.title("Hue")
    plt.xlabel("Degree 0-360")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(HSV_Value[1], bins=40, range=(0, 100), histtype='stepfilled', color='g', label='Saturation')
    plt.title("Saturation")
    plt.xlabel("Value 0-100")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(HSV_Value[2], bins=40, range=(0, 100), histtype='stepfilled', color='b', label='Value')
    plt.title("Value")
    plt.xlabel("value 0-100")
    plt.ylabel("Frequency")

    plt.legend()
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Calculate the Mean and Median HSV of the Image
def mean_median_value(im_hsv):
    # get the mean value
    hue_mean = statistics.mean(im_hsv[0])
    saturation_mean = statistics.mean(im_hsv[1])
    value_mean = statistics.mean(im_hsv[2])
    # get the median value
    h,s,l=[],[],[]
    h = im_hsv[0]
    s = im_hsv[1]
    v = im_hsv[2]
    h.sort()
    s.sort()
    v.sort()
    length=int(len(h)/2)
    hue_median = h[length]
    saturation_median = s[length]
    value_median = v[length]
    # print('MEAN        hue: {0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_mean, saturation_mean, value_mean))
    # print('MEDIUM      hue: {0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_median, saturation_median,value_median))

    # 25% median value
    new_data_hue = h[:length]
    new_data_sat = s[:length]
    new_data_val = v[:length]
    length_25=int(length/2)

    hue_25median = new_data_hue[length_25]
    sat_25median = new_data_sat[length_25]
    val_25median = new_data_val[length_25]
    # print('25% MEDIUM  hue: {0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_25median, sat_25median,val_25median))
    # 25% mean value
    hue_25mean = statistics.mean(new_data_hue)
    sat_25mean = statistics.mean(new_data_sat)
    val_25mean = statistics.mean(new_data_val)
    # print('25% MEAN    hue :{0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_25mean, sat_25mean, val_25mean))

    # 75% median value
    new_data_hue = h[length:]
    new_data_sat = s[length:]
    new_data_val = v[length:]

    hue_75median = new_data_hue[length_25]
    sat_75median = new_data_sat[length_25]
    val_75median = new_data_val[length_25]
    # print('75% MEDIUM  hue: {0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_75median, sat_75median,val_75median))
    # 75% mean value
    hue_75mean = statistics.mean(new_data_hue)
    sat_75mean = statistics.mean(new_data_sat)
    val_75mean = statistics.mean(new_data_val)
    # print('75% MEAN    hue :{0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_75mean, sat_75mean, val_75mean))
    # print('')
    # print('')



def Calculate_HSV(cropped):
    weight = cropped.shape[0]
    height = cropped.shape[1]
    lst_hue, lst_sat, lst_val = [], [], []
    for w in range(weight):
        for h in range(height):
            RGB = cropped[w, h]
            B, G, R = RGB
            # print("R:{0} G:{1} B:{2}".format(R, G, B))
            B = float(B)
            G = float(G)
            R = float(R)

            C_low = float(min(RGB))
            C_high = float(max(RGB))
            C_rng = float(C_high - C_low)
            C_max = float(255)
            # find hue
            if C_low == C_high:
                H_hsv_degree = 0
            else:
                R_prime = (C_high - R) / C_rng
                G_prime = (C_high - G) / C_rng
                B_prime = (C_high - B) / C_rng

                if C_high == R:
                    H_prime = B_prime - G_prime
                elif C_high == G:
                    H_prime = R_prime - B_prime + 2
                elif C_high == B:
                    H_prime = G_prime - R_prime + 4
                if H_prime < 0:
                    H_hsv = (H_prime + 6) / 6
                else:
                    H_hsv = H_prime / 6
                H_hsv_degree = round(360 * H_hsv, 3)

            # find saturation and Value

            V = C_high / C_max * 100
            if C_low == C_high:
                Saturation = int(0)
            else:
                Saturation = C_rng / C_high * 100

            lst_hue.append(H_hsv_degree)
            lst_sat.append(Saturation)
            lst_val.append(V)
    HSV_val = [lst_hue, lst_sat, lst_val]
    return HSV_val


def ONLY_ONE_COIN_INPUT_FOR_COLOR_CLASSIFICATION(imagePath):

    Image = cv2.imread(imagePath)
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    # INPAINT
    mask1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    result1 = cv2.inpaint(Image, mask1, 0.1, cv2.INPAINT_TELEA)
    region1 = result1[450:600, 700:850]  # y: y+h, x:x+w this is above the data
    #    y1  y2  x1   x2
    region2 = result1[400:500, 110:260]  # y: y+h, x:x+w this is above the data
    #    y1  y2  x1   x2
    region3 = result1[620:720, 130:240]  # y: y+h, x:x+w this is above the data
    #    y1  y2  x1   x2
    region4 = result1[310:380, 120:280]  # y: y+h, x:x+w this is above the data
    #    y1  y2  x1   x2
    region5 = result1[310:380, 700:880]  # y: y+h, x:x+w this is above the data
    #    y1  y2  x1   x2
    region6 = result1[220:600, 670:730]  # y: y+h, x:x+w this is above the data

    HSV_Value_r1 = Calculate_HSV(region1)  # get HSV value
    HSV_Value_r2 = Calculate_HSV(region2)  # get HSV value
    HSV_Value_r3 = Calculate_HSV(region3)  # get HSV value
    HSV_Value_r4 = Calculate_HSV(region4)  # get HSV value
    HSV_Value_r5 = Calculate_HSV(region5)  # get HSV value
    HSV_Value_r6 = Calculate_HSV(region6)  # get HSV value

    Median_S = statistics.median(
        HSV_Value_r1[1] + HSV_Value_r2[1] + HSV_Value_r3[1] + HSV_Value_r4[1] + HSV_Value_r5[1] + HSV_Value_r6[1])
    Mean_S = statistics.median(
        HSV_Value_r1[1] + HSV_Value_r2[1] + HSV_Value_r3[1] + HSV_Value_r4[1] + HSV_Value_r5[1] + HSV_Value_r6[1])
    Mean_V = statistics.median(
        HSV_Value_r1[2] + HSV_Value_r2[2] + HSV_Value_r3[2] + HSV_Value_r4[2] + HSV_Value_r5[2] + HSV_Value_r6[2])
    Mean_H = statistics.median(
        HSV_Value_r1[0] + HSV_Value_r2[0] + HSV_Value_r3[0] + HSV_Value_r4[0] + HSV_Value_r5[0] + HSV_Value_r6[0])
    list_PCA = [round(Mean_V, 1), round(Mean_S, 1), round(Mean_H, 1)]




    Coin_135 = [[34.9, 37.2, 28.3], [27.5, 48.1, 28.5], [41.2, 45.8, 26.4], [36.9, 60.8, 30.6], [45.1, 45.5, 32.3],
                [43.9, 52.5, 29.3], [36.9, 52.7, 31.6], [43.1, 49.6, 30.0], [36.9, 61.5, 30.5], [45.1, 47.4, 26.7],
                [42.7, 60.7, 32.4], [40.4, 57.7, 30.9], [49.4, 48.9, 29.1], [42.0, 49.0, 31.1], [33.7, 43.7, 25.5],
                [36.5, 42.1, 29.3], [42.0, 47.5, 30.4], [36.1, 39.8, 26.8], [35.3, 53.0, 28.5], [41.2, 46.2, 30.5],
                [38.4, 38.5, 29.1], [28.2, 64.9, 30.0], [42.0, 51.5, 30.0], [31.4, 60.8, 28.2], [35.3, 37.4, 22.7],
                [27.5, 46.8, 30.0], [40.0, 55.3, 30.6], [38.8, 43.1, 31.1], [45.9, 45.9, 27.1], [26.3, 25.0, 24.0],
                [45.9, 45.9, 27.1], [35.7, 46.0, 29.0], [51.8, 31.9, 31.3], [42.4, 42.0, 29.4], [38.4, 39.5, 35.2],
                [49.4, 41.5, 23.5], [62.7, 57.8, 22.3], [69.4, 71.4, 27.8], [45.1, 46.7, 27.7], [48.2, 73.5, 28.0],
                [53.3, 54.5, 27.1], [48.2, 52.0, 19.1], [53.3, 57.7, 23.4], [53.3, 56.2, 25.3], [45.5, 64.6, 26.4],
                [39.2, 74.4, 23.4], [45.5, 64.6, 26.4], [60.4, 46.8, 26.3], [51.8, 57.9, 25.6], [42.7, 67.6, 24.7],
                [36.1, 43.7, 35.4], [42.7, 59.8, 29.1], [45.1, 71.6, 27.0], [45.1, 62.0, 29.6], [29.0, 41.7, 21.3],
                [63.1, 52.5, 28.2], [58.8, 42.5, 20.0], [55.3, 45.9, 21.7], [76.9, 46.7, 33.6], [70.2, 61.1, 27.0],
                [58.4, 48.8, 27.1], [61.6, 56.8, 29.6], [52.5, 64.9, 29.0], [76.9, 80.1, 25.5], [78.4, 49.3, 30.0],
                [60.8, 65.1, 23.4], [43.5, 58.6, 29.3], [64.3, 69.0, 25.9], [77.3, 68.4, 30.0], [43.5, 63.1, 27.5],
                [44.7, 52.5, 28.0], [54.9, 57.1, 28.2], [87.8, 66.0, 25.8], [41.2, 78.7, 22.4], [62.7, 55.4, 27.3],
                [42.7, 46.5, 27.7], [73.3, 58.8, 29.2], [73.3, 63.6, 29.1], [66.3, 76.7, 26.5], [74.9, 54.7, 26.5],
                [84.7, 58.7, 30.3], [55.3, 64.6, 28.2], [51.0, 69.3, 23.8], [55.7, 77.7, 27.8], [60.4, 61.6, 28.0],
                [56.1, 74.9, 26.5], [85.5, 65.7, 26.2], [72.9, 62.3, 27.2], [82.4, 69.2, 26.5], [58.0, 66.0, 26.2],
                [81.2, 62.7, 29.8], [82.0, 80.1, 29.2], [70.6, 62.6, 25.6], [60.0, 65.6, 22.4], [62.4, 67.7, 26.5],
                [58.8, 72.5, 26.5], [71.0, 71.6, 26.7], [68.2, 65.5, 27.9], [79.6, 60.4, 31.6], [78.8, 74.8, 29.3],
                [58.8, 61.6, 25.1], [78.4, 78.7, 25.3], [83.5, 71.1, 28.5], [81.6, 65.5, 26.1], [60.8, 66.7, 24.7],
                [60.8, 51.2, 22.2], [75.3, 67.3, 28.0], [60.0, 57.0, 23.9], [80.4, 79.1, 23.3], [48.6, 73.8, 22.4],
                [67.8, 88.9, 24.6], [72.2, 60.6, 27.2], [67.8, 60.5, 24.9], [57.6, 67.0, 22.6], [55.7, 74.0, 26.5],
                [72.5, 75.9, 24.9], [87.8, 58.1, 32.2], [71.0, 57.3, 29.1], [63.9, 62.7, 23.8], [51.0, 72.1, 28.8],
                [83.1, 80.4, 20.4], [52.5, 60.5, 24.8], [60.0, 61.3, 23.9], [69.0, 60.2, 26.5], [71.4, 69.2, 24.2],
                [71.8, 84.8, 23.1], [87.8, 81.3, 26.9], [72.2, 71.4, 23.1], [80.4, 79.1, 23.3], [55.7, 73.7, 23.5],
                [67.8, 88.9, 24.6], [38.4, 45.2, 31.2], [42.7, 75.2, 36.3], [72.2, 70.6, 30.3], [56.1, 88.9, 24.3]]

    Coin_135.append(list_PCA)
    colors_Label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                    1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                    1, 2, 2]  # 66#tail65        #tail60

    color_list = []
    for file in colors_Label:
        if file == 2:
            color_list.append("red")
        elif file == 1:
            color_list.append("orange")
        elif file == 0:
            color_list.append("brown")


    X_A = np.array(Coin_135)
    np.cov(X_A.T)
    Color_Label = np.array(colors_Label)
    X_B = X_A

    eigvalue, eigvector = np.linalg.eig(np.cov(X_A.T))

    a = np.hstack((eigvector[:, 0].reshape(3, -1), eigvector[:, 1].reshape(3, -1)))
    X_A = X_A - X_A.mean(axis=0)
    X_new1 = X_A.dot(a)

    positive = np.ones((136, 2))
    positive = positive * 200
    X = X_new1 + positive
    y = Color_Label
    # Training Set
    X_train = X[:135, :]
    y_train = y[:]
    # Test Set
    X_test = X[135:136, :]
    KNN = KNearestNeighbor()
    KNN.train(X_train, y_train)
    y_pred = KNN.predict(X_test, k=5)
    results = " "
    if y_pred == 0:
        results = "Brown"
    elif y_pred == 1:
        results = "Red-Brown"
    else:
        results = "Red"
    # ----------Outliers-------------START
    outliers = 0

    X_test=X[135:136, :]
    for i in range(10):
        X1 = X[134 - i:135 - i, :]
        dist = np.sqrt(np.sum(np.square(X_test - X1)))
        if dist < 1:
            if i == 0 or i == 5:
                outliers = 1#"Red Maybe"
            elif i == 2:
                outliers = 1#"Colorful"
            elif i == 3:
                outliers = 1#"Cleaned"
            elif i == 4:
                outliers = 1#"Pretty toning"
            elif i == 6:
                outliers = 1#"Great Luster"
            elif i == 8:
                outliers = 1#"MS67"
            elif i == 9:
                outliers = 1#"Questionable"
            elif i == 7:
                outliers = 1#"Cheery red"
            elif i == 1:
                outliers = 1#"golden"
            break
        else:
            outliers = 0
    if(outliers):
        results2="; Possible Outlier"
    else:
        results2=" "


    # -------------Outliers-------------END
    Final_Output=[results,results2]
    return Final_Output






class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))  
        
        d1 = -2 * np.dot(X, self.X_train.T) 
        d2 = np.sum(np.square(X), axis=1, keepdims=True)  
        d3 = np.sum(np.square(self.X_train), axis=1)  
        dist = np.sqrt(d1 + d2 + d3)

        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dist_k_min = np.argsort(dist[i])[:k] 
            y_kclose = self.y_train[dist_k_min]  
            y_pred[i] = np.argmax(np.bincount(y_kclose))  

        return y_pred



def Image_HSV_Region1(imagePath):
    list_RP = []
    list_Median_H = []
    list_Median_S = []
    list_Median_V = []
    list_NRP = []
    #imagePath= findImagePath()
    print(imagePath)
    #img = loadImages('color', imagePath)
    img = []
    img.append(cv2.imread(imagePath, cv2.IMREAD_COLOR))
    #img.append(imagePath)
    #print(img)
    i = 0
    print("REGION1")
    for Image in img:
         # cropped=Image[300:400,300:500]  #this is for RGB

         gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
         # INPAINT
         mask1 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
         result1 = cv2.inpaint(Image, mask1, 0.1, cv2.INPAINT_TELEA)


         region1 = result1[450:600,700:850]  # y: y+h, x:x+w this is above the date
                    #    y1  y2  x1   x2
         region2 = result1[400:500,110:260]  # y: y+h, x:x+w this is above the liberty
                    #    y1  y2  x1   x2
         region3 = result1[620:720,130:240]  # y: y+h, x:x+w this is below the liberty
                    #    y1  y2  x1   x2
         region4 = result1[310:380,120:280]  # y: y+h, x:x+w this is below the "in god"
                    #    y1  y2  x1   x2
         region5 = result1[310:380,700:880]  # y: y+h, x:x+w this is in front of lincoln's eyes
                    #    y1  y2  x1   x2
         region6 = result1[220:600,670:730]  # y: y+h, x:x+w this is below "trust"
                    #    y1  y2  x1   x2
    

        # SHOW THE RED REGIONS OF THE COIN BEING SCANNED
         """
         # above the date
         cv2.rectangle(Image, (700, 450), (850, 600), (0, 250, 0), 3)   # x left y top  x right  y bottom
        # above liberty 
         cv2.rectangle(Image, (110, 400), (260, 500), (0, 0, 250), 3)   # x left y top  x right  y bottom
                              # x1   y1    x2   y2
        # below liberty
         cv2.rectangle(Image, (130, 620), (240, 720), (0, 0, 250), 3)   # x left y top  x right  y bottom
                              # x1   y1    x2   y2
        # below "in god"
         cv2.rectangle(Image, (120, 310), (280, 380), (0, 0, 250), 3)   # x left y top  x right  y bottom
                              # x1   y1    x2   y2
        # lincoln's eyes
         cv2.rectangle(Image, (700, 310), (880, 380), (0, 0, 250), 3)   # x left y top  x right  y bottom
                              # x1   y1    x2   y2
        # below "trust"
         cv2.rectangle(Image, (670, 220), (730, 290), (0, 0, 250), 3)   # x left y top  x right  y bottom


         cv2.imshow('image',Image)
         plt.show()
         cv2.waitKey(0)
         cv2.destroyAllWindows()
         """

         HSV_Value_r1=Calculate_HSV(region1)  #get HSV value
         HSV_Value_r2=Calculate_HSV(region2)  #get HSV value
         HSV_Value_r3=Calculate_HSV(region3)  #get HSV value
         HSV_Value_r4=Calculate_HSV(region4)  #get HSV value
         HSV_Value_r5=Calculate_HSV(region5)  #get HSV value
         HSV_Value_r6=Calculate_HSV(region6)  #get HSV value
         print("---------------------------")

         """
         Histogram_drawing(HSV_Value_r1)
         Histogram_drawing(HSV_Value_r2)
         Histogram_drawing(HSV_Value_r3)
         Histogram_drawing(HSV_Value_r4)
         Histogram_drawing(HSV_Value_r5)
         Histogram_drawing(HSV_Value_r6)
         """
         

         mean_median_value(HSV_Value_r1)      #print mean median
         mean_median_value(HSV_Value_r2)      #print mean median
         mean_median_value(HSV_Value_r3)      #print mean median
         mean_median_value(HSV_Value_r4)      #print mean median
         mean_median_value(HSV_Value_r5)      #print mean median
         mean_median_value(HSV_Value_r6)      #print mean median

         All_H = HSV_Value_r1[0] + HSV_Value_r2[0] + HSV_Value_r3[0] + HSV_Value_r4[0] + HSV_Value_r5[0] + HSV_Value_r6[0]
         All_S = HSV_Value_r1[1] + HSV_Value_r2[1] + HSV_Value_r3[1] + HSV_Value_r4[1] + HSV_Value_r5[1] + HSV_Value_r6[1]
         All_V = HSV_Value_r1[2] + HSV_Value_r2[2] + HSV_Value_r3[2] + HSV_Value_r4[2] + HSV_Value_r5[2] + HSV_Value_r6[2]
         
         Median_H = statistics.mean(All_H)
         Median_S=statistics.median(All_S)
         Median_V=statistics.median(All_V)

         print("Median: ", Median_S)

         Min_S =min(All_S)
         Min_V =min(All_V)

         Max_S =max(All_S)
         Max_V =max(All_V)

         # Old Percentage Red Equation
         """
         Pecetange_Red = (((Median_S+Median_V)-(Min_S+Min_V))/((Max_S+Max_V)-(Min_S+Min_V)))*100
         Pecetange_Red =round(Pecetange_Red,1)

         list_RP.append(Pecetange_Red)
         """
         New_Percentage_Red = ((2*Median_S * Median_V)/(Median_S + Median_V)) * math.exp(-((Median_H - 26)/8)**2)
         New_Percentage_Red =round(New_Percentage_Red,1)
         list_NRP.append(New_Percentage_Red)
         list_Median_H.append(Median_H)
         list_Median_S.append(Median_S)
         list_Median_V.append(Median_V)
         i+=1
    
    # calculate the %Red of coin
    # print("Percentage: ")
    # print(list_RP)

    print("New Percentage: ")
    print(list_NRP)
    print("Median Hue: ")
    print(list_Median_H)
    print("Median Sat: ")
    print(list_Median_S)
    print("Median Val: ")
    print(list_Median_V)

    # print('Possible Red Percentage: {0:.1f}'.format(Pecetange_Red))

    print('Possible New Red Percentage: {0:.1f}'.format(New_Percentage_Red))

    BN=[37.234042553191486, 48.148148148148145, 45.794392523364486, 60.824742268041234, 47.482014388489205, 57.65765765765766, 48.854961832061065, 43.67816091954023, 42.10526315789473, 38.53211009174312, 45.87155963302752, 45.87155963302752, 31.88405797101449]
    QC=[84.78]
    RB=[42.05607476635514, 58.333333333333336, 71.56180703508848, 46.666666666666664, 73.52941176470588, 54.47154471544715, 52.02702702702703, 57.66647261654202, 56.209150326797385, 64.58333333333334, 74.35897435897436, 64.58333333333334, 46.846846846846844, 57.943925233644855, 67.56756756756756, 43.67816091954023, 59.756097560975604, 71.55963302752293, 61.95652173913043, 41.66666666666667, 52.55474452554745, 42.51497005988024, 45.87155963302752]
    RED=[47.95918367346938, 61.37566137566137, 49.142857142857146, 57.02479338842975, 64.90066225165563, 80.0925925925926, 49.45652173913043, 65.16129032258064, 58.58585858585859, 69.06474820143885, 68.42105263157895, 63.10679611650486, 52.54237288135594, 57.22543352601156, 66.21004566210046, 78.70370370370371, 55.35714285714286, 46.478873239436616, 59.03083700440529, 63.69426751592356, 76.66666666666667, 55.0561797752809, 60.451977401129945, 64.65517241379311, 69.2982456140351, 77.6978417266187, 61.63522012578616, 74.87179487179488, 65.93886462882097, 62.841530054644814, 69.47368421052632, 65.98639455782312, 63.687150837988824, 80.12422360248446, 62.58064516129033, 65.625, 67.72486772486772, 72.53521126760563, 71.58469945355192, 65.63876651982379, 61.66666666666667, 74.75728155339806, 61.73913043478261, 78.82882882882883, 71.07843137254902, 65.51724137931035, 66.66666666666666, 51.24999999999999, 67.5, 57.03703703703704, 79.07949790794979, 73.80952380952381, 88.88888888888889, 60.95890410958904, 60.547338201335585, 67.04545454545455, 74.0, 75.88235294117646, 58.29383886255924, 58.00865800865801, 62.70270270270271, 72.11538461538461, 80.42328042328042, 60.46511627906976, 61.25000000000001, 60.73059360730594, 69.36416184971098]
    STEEL = [3.896103896103896, 4.504504504504505, 7.8431372549019605, 5.0, 9.523809523809524, 4.580152671755725, 2.9850746268656714]
    LR = [81.28]
    proto = [52.42718446601942, 62.0253164556962]

    BN_P=[35.7, 39.7, 42.2, 44.6, 43.7, 47.7, 45.1, 36.7, 40.7, 38.2, 39.7, 39.7, 45.6]
    QC_P=[69.9]
    RB_P=[43.4, 58.4, 62.6, 44.5, 57.0, 43.1, 46.0, 50.3, 49.0, 55.1, 53.3, 55.1, 46.9, 45.6, 51.6, 36.7, 49.4, 53.1, 45.5, 33.5, 51.5, 45.9, 46.9]
    RED_P=[56.4, 62.9, 53.8, 55.5, 57.2, 67.1, 53.6, 59.6, 42.7, 60.9, 59.8, 46.3, 43.3, 48.6, 75.2, 45.8, 50.9, 34.4, 57.1, 60.8, 62.4, 57.3, 65.7, 54.5, 56.5, 62.1, 54.8, 50.8, 70.8, 57.5, 68.0, 47.8, 67.8, 74.4, 54.0, 51.5, 57.4, 54.6, 60.6, 59.8, 63.9, 65.2, 50.5, 72.2, 69.0, 63.7, 59.2, 47.2, 63.4, 45.7, 70.5, 49.4, 67.8, 62.5, 55.1, 53.7, 55.3, 68.0, 61.1, 56.4, 51.8, 54.8, 73.5, 49.8, 47.8, 58.8, 65.3]
    STEEL_P = [57.0, 31.7, 47.9, 65.1, 44.1, 35.8, 45.0]
    LR_P = [72.9]

    N_QC_P=[69.9]
    N_BN_P=[25.0, 32.3, 43.0, 31.9, 45.2, 31.8, 41.5, 38.1, 32.7, 33.2, 44.5, 44.5, 21.7]
    N_RB_P=[41.0, 51.0, 65.8, 43.1, 54.5, 52.7, 23.8, 50.5, 54.1, 53.2, 45.4, 53.2, 52.7, 54.6, 50.8, 6.9, 44.0, 54.3, 43.0, 28.4, 53.6, 28.5, 36.3]
    N_RED_P=[26.7, 64.8, 45.7, 48.7, 50.2, 78.4, 48.3, 57.4, 41.0, 66.4, 57.8, 49.6, 44.7, 52.6, 74.5, 43.6, 56.9, 43.2, 56.6, 59.3, 70.9, 63.2, 55.4, 55.4, 54.3, 61.7, 57.2, 63.8, 74.2, 66.4, 74.6, 61.7, 59.0, 68.4, 66.2, 50.7, 64.7, 64.3, 70.7, 63.7, 45.0, 65.3, 59.4, 78.6, 70.2, 72.6, 62.1, 43.5, 67.1, 54.2, 75.4, 48.4, 73.1, 65.0, 62.9, 52.0, 63.1, 73.2, 39.6, 56.0, 58.6, 52.0, 52.4, 54.3, 56.0, 64.4, 67.1]
    N_STEEL_P = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    N_LR_P = [82.6]
    proto_p = [39.4, 58.5]

    overalP = BN_P + RB_P + RED_P + QC_P + LR_P 
    overallX = BN + RB + RED + QC + LR 

    #prototyping
    #overalP = proto_p
    #overallX = proto

    # %Red Plot for Old Equation
    """
    x = np.array(overallX)
    m, b = np.polyfit(overallX, overalP, 1)
    print('m=',m)
    print('b=',b)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.scatter(BN,BN_P,  c ='brown')        # plot x and y using default line style and color
    plt.scatter(RB, RB_P,  c ='orange')
    plt.scatter(RED, RED_P,  c ='red')
    plt.scatter(QC, QC_P,  c ='purple')
    plt.scatter(STEEL, STEEL_P,  c ='black')
    plt.scatter(LR, LR_P, c="green")
    plt.plot(x, m * x + b)
    plt.title("Percentage of Red vs Average Saturation")
    plt.ylabel("Percentage of Red")
    plt.xlabel("Average Saturation")
    plt.show()
    """
    

    overallNewP = N_BN_P + N_RB_P + N_RED_P + N_QC_P + N_LR_P
    #prototyping
    #overallNewP = proto_p

    # %Red Plot for New Equation
    """
    x = np.array(overallX)
    m, b = np.polyfit(overallX, overallNewP, 1)
    
    print('m=',m)
    print('b=',b)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.scatter(BN,N_BN_P,  s=80, c ='brown')        # plot x and y using default line style and color
    plt.scatter(RB, N_RB_P, s=80, c ='orange')
    plt.scatter(RED, N_RED_P, s=80, c ='red')
    plt.scatter(QC, N_QC_P, s=80, c ='purple')
    plt.scatter(STEEL, N_STEEL_P,s=80,  c ='black')
    plt.scatter(LR, N_LR_P, s=80, c="red")
    #prototyping
    #plt.scatter(proto[0], proto_p[0], c="brown")
    #plt.scatter(proto[1], proto_p[1], c="red")
    plt.plot(x, m * x + b)
    plt.title("Percentage of Red vs Average Saturation", fontsize=15)
    plt.ylabel("Percentage of Red", fontsize=15)
    plt.xlabel("Average Saturation", fontsize=15)
    plt.show()
    
    """
    
    print("------------------------------------")
    print("Percentage: ", New_Percentage_Red)
    return New_Percentage_Red


def isMSD(imagePath):
    percentage = Image_HSV_Region1(imagePath)
    if percentage < 3:
        return True
    return False

if __name__ == '__main__':
    Image_HSV_Region1()
 
