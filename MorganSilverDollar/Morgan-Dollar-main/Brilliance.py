"""
Brilliance.py

Purpose: The purpose of this file is to calculate the brilliance of a coin.
The brilliance of a coin tells the user how shinny or dull the coin is.
If the coin is very shinny/brilliant coin will receive a 10 on the scale, but
if the coin is dull it will receive a 1.

Explanation of code: The code first calculates the HSV with function Calculate_HSV,
then uses the value that was calculated for the next function getBrilliance_ And_Percent_Silver
and receive a brilliance scale value and percent silver value. The reason the value starts
at 7.8 for most brilliance is because when we looked up what the pure silver value
would be for that color it was %75, so I made it a little higher because that would mean the coin
is extremely brilliant. Any higher and the coin is more like a white
coin which means it is extremely brilliant, which meets our requirements. Brilliance is just
telling us the shin and percent silver tells us how much of the coin is pure silver.

How to run File: Have an arbitrary value equal getBrilliance with an image path of
the users choice plugged in. Then, print that arbitrary value, and the user will be
able to see what brilliance the image of their choice gives.

Author: Zymmorrah Myers
Date: 12 Mar 2023
"""
import cv2
from ImageHSV import statistics
from CoinImage import CoinImage
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the necessary libraries
from PIL import Image

# Received from HSV file that was provided from the previous coin team
def Calculate_HSV(coin):
    """
     * In the function below RGB is converted to HSV using a set of formulas.  The
     * HSV values for each pixel within the image is calculated starting from (x=0, y=0) to
     * (x=width-1, y=height-1)
     *
     * Param:   coin - CoinImage object comprising the BGR image
     *
     * Return:  HSV_value - A list of 3 lists, the first list contains all hue values, the second contains all
     *                      saturation values, and the third contains all value values.
    """
    # image you want the HSV of
    image = coin.colorimg

    # Obtains the width and height of the region to run calculations on
    width = image.shape[0]
    height = image.shape[1]
    # Initialization of lists for hue, saturation, and value values
    lst_hue, lst_sat, lst_val = [], [], []
    # Read RGB for every pixel

    for w in range(width):
        for h in range(height):
            RGB = image[w, h]
            # When read from an image, RGB is read in the order of Blue, Green, Red
            B, G, R = RGB
            B, G, R = float(B), float(G), float(R)
            # This will skip over any black pixels (mask pixels)
            if R == 0 and G == 0 and B == 0:
                continue
            # finds the smallest value between Red, Green, and Blue
            C_low = float(min(RGB))
            # finds the largest value between Red, Green, and Blue
            C_high = float(max(RGB))
            # finds the difference between the max and min
            C_range = float(C_high - C_low)
            """
             * Just as a note for visualization: 
             *    - Hue is the angle on the color wheel
             *    - Saturation is the distance from the center of the color wheel (further is greater saturation)
             *    - Value is the depth within the color wheel
            """
            # If there is not max or min, then the calculations are unnecessary
            if C_low == C_high:
                H_hsv_degree = 0
            else:
                H_prime = 0

                # Formula for when Red is the highest
                if C_high == R:
                    # H_prime = np.divide(np.add(G, B), C_range)
                    H_prime = (G - B) / C_range
                # Formula for when Green is the highest
                elif C_high == G:
                    H_prime = 2.0 + (B - R) / C_range
                # Formula for when Blue is the highest
                elif C_high == B:
                    H_prime = 4.0 + (R - G) / C_range
                """
                 * Multiply result by 60 to scale to 360 degrees.  If the result was negative, add 360 to get the 
                 * equivalent position angle.
                """
                if H_prime < 0:
                    H_hsv = (H_prime * 60) + 360
                else:
                    H_hsv = H_prime * 60

                H_hsv_degree = round(H_hsv, 3)

            # find saturation and Value
            C_high_bin = C_high / 255
            C_range_bin = C_range / 255
            # Find the value, the value is determined by the color with the greatest strength in RGB
            V = round(C_high_bin * 100, 3)
            # Find the saturation by applying a conversion formula, if all RGB values are the same then there is no sat
            if C_low == C_high:
                Saturation = int(0)
            else:
                Saturation = round((C_range_bin / C_high_bin) * 100, 3)
            lst_hue.append(H_hsv_degree)
            lst_sat.append(Saturation)
            lst_val.append(V)
    # Creates a new list composed of all HSV values
    HSV_val = [lst_hue, lst_sat, lst_val]

    # Creates the median of the hue, sat, and val
    mean_hue = np.mean(lst_hue)
    mean_sat = np.mean(lst_sat)
    mean_val = np.mean(lst_val)
    return mean_hue, mean_sat, mean_val


# Calculates the Percent Silver and brilliance of each channel of the image at the given path.
def getBrilliance_And_Percent_Silver(coin):
    H, S, V = Calculate_HSV(coin)

    # Previous equation used: (worked as well)New_Percentage_Silver = ((((100 - hsv_medians[1])*hsv_medians[2])/(hsv_medians[1] + hsv_medians[2]))* 0.01)
    sat = float((50 - S) * 2)
    og_val = float(V)

    # The following will determine the percent using Saturation and Value to determine the percent of silver in a none toned coin
    # This if statement is done so that the values are even and a 100% is reachable
    if og_val <= 75:
        # value part 1 between 0 - 75
        v_p1 = og_val
        # value part 2 between 0 - 25
        v_p2 = og_val * 0.3333
        # official value
        val = v_p1 + v_p2
    else:
        off_set = og_val - 75
        # value part 1 between 0 - 75 (gives how much 75 was off set from the beginning)
        v_p1 = 75 - off_set
        # value part 2 between 0 - 25
        v_p2 = v_p1 * 0.3333
        # official value
        val = v_p1 + v_p2
    New_Percentage_Silver = float(((2 * sat * val) / (sat + val)))
    New_Percentage_Silver = round(New_Percentage_Silver, 2)

    New_Brilliance_Degree = round((og_val / 10), 2)
    if New_Brilliance_Degree >= 7.8:
        New_Brilliance_Degree = 10.0
    elif 7.8 > New_Brilliance_Degree > 7.25:
        New_Brilliance_Degree = 9.0
    elif 7.25 > New_Brilliance_Degree > 7:
        New_Brilliance_Degree = 8.0
    elif 7 > New_Brilliance_Degree > 6.75:
        New_Brilliance_Degree = 7.0
    elif 6.75 > New_Brilliance_Degree > 6.5:
        New_Brilliance_Degree = 6.0
    elif 6.5 > New_Brilliance_Degree > 6.25:
        New_Brilliance_Degree = 5.0
    elif 6.25 > New_Brilliance_Degree > 6:
        New_Brilliance_Degree = 4.0
    elif 6 > New_Brilliance_Degree > 5.75:
        New_Brilliance_Degree = 3.0
    elif 5.75 > New_Brilliance_Degree > 5.5:
        New_Brilliance_Degree = 2.0
    elif 5.5 > New_Brilliance_Degree > 0:
        New_Brilliance_Degree = 1.0

    # The following prints the Percent Silver, Brilliance Degree, Median Hue, Median Saturation, and Median Val of
    # each channel
    print("Percentage: " + str(New_Percentage_Silver))
    print("Brilliance: " + str(New_Brilliance_Degree))
    print("Mean Hue: " + str(H))
    print("Mean Sat: " + str(S))
    print("Mean Val: " + str(V))

    # Generate data on commute times.
    """size, scale = 5000, 10
    commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)

    commutes.plot.hist(grid=True, bins=1, rwidth=0.9,
                       color='#607c8e')
    plt.title('Brightness Scores for 4939 coins')
    plt.xlabel('Amount of Coins')
    plt.ylabel('Scores (1-10)')
    plt.grid(axis='y', alpha=0.75)
    histogram = pd.DataFrame.histogram()"""
    return New_Brilliance_Degree, New_Percentage_Silver

def getBrillianceHist():
    "Only ussing obverse image because the graphs for both reverse and obverse are the same"
    d = pd.read_csv("image_database.csv", usecols=['Brilliance Obverse'])
    d.hist(bins = 20)
    plt.xlim([1, 10])
    plt.ylim([0, 4000])
    plt.title("Brilliance Histogram")
    plt.xlabel("Brilliance Scores (1 - 10)")
    plt.ylabel("Number of Coins")
    histPlot = plt.figure(1)
    histPlot.savefig('BrillianceHist.jpg')
    img = Image.open('BrillianceHist.jpg')
    Hist = np.array(img)

    return Hist

if __name__ == '__main__':
    getBrillianceHist()


