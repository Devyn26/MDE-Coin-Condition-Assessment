"""
ConditionFeaturesEval.py

~ MaskImages are binary - white (255) in the region of interest, black otherwise

Original Author: Creed Jones
Date: 8 Sep 2022
Modified By: Lizzie LaVallee, Jasper Emick
Date: 10 Mar 2023
"""
import math
import numpy as np
import cv2
import os


class MaskImage:  # a mask is all black, except white within a subregion defining the mask
    # the mask is used as an area to operate within
    # currently, only wedge-shaped masks are defined
    def __init__(self, rows=1000, cols=1000):
        self.M = np.zeros((rows, cols), dtype=np.uint8)
        self.points = None

    def clear(self):
        self.M.fill(0)

    def math_angle_to_nautical(self, ang):  # takes a math angle in radians and converts to nautical in degrees
        result = 90 + (180.0 / np.pi) * ang
        while (result > 360):
            result -= 360
        while (result < 0):
            result += 360
        return result

    def nautical_angle_to_math(self, ang):  # takes a nautical angle in degrees and converts to math, in radians
        result = (np.pi / 180.0) * (-ang + 90)
        while (result > np.pi):
            result -= 2 * np.pi
        while (result < -np.pi):
            result += 2 * np.pi
        return result

    def create_wedge(self, start_angle, end_angle, wedgeRadius, xc, yc):
        if wedgeRadius != 0:
            start_math_angle = self.nautical_angle_to_math(start_angle)
            end_math_angle = self.nautical_angle_to_math(end_angle)
            start_angle_point = (
                int(wedgeRadius * math.cos(start_math_angle) + xc),
                int(yc - wedgeRadius * math.sin(start_math_angle)))
            end_angle_point = (int(wedgeRadius * math.cos(end_math_angle) + xc),
                               int(yc - wedgeRadius * math.sin(end_math_angle)))


            wedgeMask = np.zeros( (1000, 1000), dtype=np.uint8)
            #self.points = np.asarray([(xc, yc), start_angle_point, end_angle_point])
            wedge = cv2.rectangle(img=wedgeMask, pt1=start_angle_point, pt2=end_angle_point, color=(255), thickness=-1)
        else:
            wedge = np.ones( (1000, 1000), dtype=np.uint8) * 255

        # USE mask_edges, KEEP THIS FUNCTION JUST FOR WEDGES.
        # # Initialize a new mask as an empty image
        maskCircle = np.zeros_like(self.M)
        # # Create a circular mask to apply to each image
        maskCircle = cv2.circle(maskCircle, (xc, yc), (int(wedgeRadius)), (255, 255, 255), -1)
        # # Applies the circular mask to the image
        maskedImage = cv2.bitwise_and(maskCircle, wedge)
        self.M = maskedImage
       #self.M = wedge

    def read_rectangle(self, topLeft, botRight):
        """
            Read a rectangle of the coin
        """
        maskRect = np.zeros_like(self.M)
        maskRect = cv2.rectangle(img=maskRect, pt1=topLeft, pt2=botRight, color=(255), thickness=-1)
        self.M = maskRect

    def pie_slice(self, start_angle=0, end_angle=45, wedgeRadius=3000, xc=500, yc=500):
        """
            Create a pie slice / wedge mask to read from angle start_angle to angle end_angle. 0 is straight up,
            positive is clockwise
        """
        if wedgeRadius != 0:
            start_math_angle = self.nautical_angle_to_math(start_angle)
            end_math_angle = self.nautical_angle_to_math(end_angle)
            start_angle_point = (
                int(wedgeRadius * math.cos(start_math_angle) + xc),
                int(yc - wedgeRadius * math.sin(start_math_angle)))
            end_angle_point = (int(wedgeRadius * math.cos(end_math_angle) + xc),
                               int(yc - wedgeRadius * math.sin(end_math_angle)))

            self.points = np.asarray([(xc, yc), start_angle_point, end_angle_point, (xc, yc)])

            wedgeMask = np.zeros((1000, 1000), dtype=np.uint8)
            wedge = cv2.fillPoly(img=wedgeMask, pts=[self.points], color=(255))
        else:
            wedge = np.ones((1000, 1000), dtype=np.uint8) * 255

        self.M = wedge

    def mask_edges(self, coinCenter=(500,500), radius=490):
        # Initialize a new mask as an empty image
        maskCircle = np.zeros_like(self.M)
        # Create a circular mask to apply to each image
        maskCircle = cv2.circle(maskCircle, (int(coinCenter[0]), int(coinCenter[1])), int(radius), (255, 255, 255), -1)
        # Applies the circular mask to the image
        self.M = cv2.bitwise_and(self.M, maskCircle)

    def mask_circle(self, coinCenter=(500, 500), radius=490):
        maskCircle = np.zeros_like(self.M)
        # maskCircle = cv2.bitwise_not(maskCircle)
        maskCircle = cv2.circle(maskCircle, (int(coinCenter[0]), int(coinCenter[1])), int(radius), (255, 255, 255), -1)
        maskCircle = cv2.bitwise_not(maskCircle)
        self.M = cv2.bitwise_and(self.M, maskCircle)

    def save(self, file):
        cv2.imwrite(file, self.M)

    @staticmethod
    def custom_mask(img, maskImg):
        mask = cv2.imread(maskImg, 0)
        masked_img = cv2.bitwise_and(mask, img)
        return masked_img

    def place_mask(self, coinImg):
        return cv2.bitwise_and(coinImg, self.M)
