import cv2
from cv2 import exp
from matplotlib.collections import PathCollection
import numpy
import os
from ImageOpener import loadImages
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
import imutils
from sklearn.decomposition import PCA
from GUI import *

#added pandas to export data
import pandas as pd
# added mplcursor to select each plot for data
import mplcursors
# import colorFigure to plot at the end
# import colorFigure
# import date to figure out how long the code runs
from datetime import datetime

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

def Calculate_HSV(cropped,mask):
    weight = cropped.shape[0]
    height = cropped.shape[1]
    lst_hue, lst_sat, lst_val = [], [], []
    for w in range(weight):
        for h in range(height):
            if mask[w,h]==0:
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
   
def oneCoinRed(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # INPAINT
    mask1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
    # result1 = cv2.inpaint(Image, mask1, 0.1, cv2.INPAINT_TELEA)
    region1 = image

    HSV_Value_r1=Calculate_HSV(region1,mask1)  #get HSV value
        

    mean_median_value(HSV_Value_r1)      #print mean median

    All_H = HSV_Value_r1[0]
    All_S = HSV_Value_r1[1] 
    All_V = HSV_Value_r1[2]
    Median_H = statistics.median(All_H)
    Median_S=statistics.median(All_S)
    Median_V=statistics.median(All_V)

    Min_S =min(All_S)
    Min_V =min(All_V)

    Max_S =max(All_S)
    Max_V =max(All_V)

    New_Percentage_Red = ((2*Median_S * Median_V)/(Median_S + Median_V)) * math.exp(-((Median_H - 25.5) / 13.5)**2)
    New_Percentage_Red =round(New_Percentage_Red,1)

    return New_Percentage_Red
    # print("Red%",New_Percentage_Red)

if __name__ == '__main__':
    print("Enter the coin filename: ")
    file = input()
    oneCoinRed(file)
 
