"""
CoinImage.py

Initializes an object class CoinImage for holding image data related to an individual coin image and providing
functionality for coin orientation analysis

Original Author: Creed Jones
Date: 8 Sep 2022
Modified By: Lizzie LaVallee, Jasper Emick, Phil Johnson
Date: 10 Mar 2023
"""

# Coins2022/CoinImage.py     Creed Jones     VT ECE  Sept 8, 2022
# development in support of the MDE coin grading teams AY22-23
# CoinImage objects are loaded, operated on, displayed and stored as needed
import sys

import cv2
import math
import numpy as np
import os
from skimage import morphology
from MaskImage import MaskImage


class CoinImage:
    def __init__(self):
        self.colorimg = None
        self.grayimg = None
        self.binaryimg = None
        self.stdimg = None
        self.coincenter = None
        self.coinradius = None
        self.filename = None
        self.coinangle = None  # the degrees the coin should be rotated from std position

    def load(self, fname):
        try:
            self.colorimg = cv2.imread(fname)
            try:
                self.reAdjustImage()  # resize and reposition the coin
                self.findcenter()
            except:
                self.coincenter = (500, 500)
                self.coinradius = 485
            # self.save("adjusted.jpg")
            self.grayimg = cv2.cvtColor(self.colorimg, cv2.COLOR_BGR2GRAY)
        except:
            self.colorimg = None
            self.grayimg = None
        self.filename = fname
        return self.colorimg

    def save(self, fname):
        cv2.imwrite(fname, self.colorimg)

    # circular import using ImageAdjuster.py so these are copied here until further notice :(
    @staticmethod
    def rescaleImg(img, scale, center):
        """
            resizes an image (zoom in/out)
            scale < 1 for zoom out, > 1 for zoom in
        """
        rot_mat = cv2.getRotationMatrix2D(center, 0, scale)
        result = cv2.warpAffine(img, rot_mat, (1000, 1000), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return result

    @staticmethod
    def shiftImage(img, r_shift, d_shift):
        """ Shifts img r_shift to the right and d_shift down """
        translation_matrix = np.float32([[1, 0, r_shift], [0, 1, d_shift]])
        result = cv2.warpAffine(img, translation_matrix, (1000, 1000), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        return result

    def reAdjustImage(self):
        """ Adjust this image to match the template dimensions """
        targetCenter = (500, 500)
        targetRadius = 485
        original = self.colorimg
        try:
            self.findcenter()
        except:
            self.colorimg = CoinImage.rescaleImg(self.colorimg, .97, (500, 500))
            try:
                self.findcenter()
            except:
                self.colorimg = original  # revert
                return

        # # rotate the coin
        # rotateAngle = targetAngle - currAngle
        # self.stdimg = CoinImage.rotateImg(self.stdimg, rotateAngle, self.coincenter)

        # shift position
        right_shift = targetCenter[0] - self.coincenter[0]
        down_shift = targetCenter[1] - self.coincenter[1]
        # if abs(right_shift) >= 1 or abs(down_shift) >= 1:
        self.colorimg = CoinImage.shiftImage(self.colorimg, right_shift, down_shift)

        # resize
        scale = targetRadius / self.coinradius
        # if scale < .9 or scale > 1.1:
        self.colorimg = CoinImage.rescaleImg(self.colorimg, scale, self.coincenter)
        try:
            self.findcenter()
        except:
            self.colorimg = original

        return self.colorimg

    def findcenter(self):
        self.grayimg = cv2.cvtColor(self.colorimg, cv2.COLOR_BGR2GRAY)

        inverted = cv2.bitwise_not(self.grayimg)  # invert black and white because of threshold direction

        thr, self.binaryimg = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY_INV)  # invert back in the process

        params = cv2.SimpleBlobDetector_Params()

        params.filterByConvexity = False
        params.filterByInertia = False

        params.filterByArea = True
        params.minArea = 500000  # coin must be at least half the full image size
        params.maxArea = 9999999999999
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        if self.binaryimg is None:
            return None
        keypoints = detector.detect(self.binaryimg)

        # im_with_keypoints = cv2.drawKeypoints(self.colorimg, keypoints, np.array([]), (0, 0, 255),
        #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if len(keypoints) != 0:
            self.coincenter = keypoints[0].pt
            self.coinradius = keypoints[0].size / 2
        else:
            self.coincenter = (500, 500)
            self.coinradius = 485
        return 1

    """ Find the angle of the coin's rotation **NOT WORKING!!"""
    def findangle(self):
        """
            Finds the angle of the coin using Hough transform and measuring the angle of straight lines on the coin.
            Want the neck line to be 90 degrees, nose line to be 60 degrees
        """
        def display(coin, neck, nose):
            """
                display the coin with the chosen neck and nose lines
            """
            displayImg = coin.grayimg
            if neck is not None:
                cv2.line(displayImg, (neck[0], neck[1]), (neck[2], neck[3]), (0, 0, 255), 2)
            if nose is not None:
                cv2.line(displayImg, (nose[0], nose[1]), (nose[2], nose[3]), (0, 0, 255), 2)

            cv2.namedWindow('gray with line', cv2.WINDOW_NORMAL)
            cv2.imshow('gray with line', displayImg)
            cv2.waitKey(0)

        # Get rid of as much noise as possible before detecting the lines
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
        clImg = clahe.apply(self.grayimg)
        blur = cv2.GaussianBlur(clImg, ksize=(5, 5), sigmaX=7, sigmaY=7)
        edge = self.evaluateEdge(blur)
        preprocessed = edge

        # Masks everything else to isolate the straight lines on the coin
        neck_mask = MaskImage()
        nose_mask = MaskImage()
        neck_mask.read_rectangle((325, 715), (410, 800))  # neck straight line
        nose_mask.read_rectangle((210, 430), (275, 490))  # nose straight line

        # Doesn't edit the og image, returns a copy so don't need to reload before making image adjustments later :)
        neckImg = neck_mask.place_mask(preprocessed)
        noseImg = nose_mask.place_mask(preprocessed)

        # Detect lines using hough
        neckLines = cv2.HoughLinesP(neckImg, 1, np.pi / 180, 10, minLineLength=50, maxLineGap=1)
        noseLines = cv2.HoughLinesP(noseImg, 1, np.pi / 180, 10, minLineLength=50, maxLineGap=1)

        # make into Nx4 matrix
        if neckLines is not None:
            neckLines = neckLines.reshape(len(neckLines), 4)
        if noseLines is not None:
            noseLines = noseLines.reshape(len(noseLines), 4)

        def find_longest(lines):
            """
                find the longest line in an array of lines
            """
            if lines is None:
                return None
            radius = np.zeros((lines.shape[0], 2))
            radius[:, 0] = np.subtract(lines[:, 0], lines[:, 1])
            radius[:, 1] = np.subtract(lines[:, 2], lines[:, 3])
            radius = np.power(radius, 2)
            radius = np.sum(radius, axis=1)
            radius = np.sqrt(radius)
            longest_line = lines[np.argmax(radius), :]
            return longest_line

        def find_line_angle(line):
            """
                find the angle [0, 180) of the line in degrees. (unit circle angle reference)
            """
            if line is None:
                return None
            x1, y1, x2, y2 = line
            if y1 < y2:
                x2, y2, x1, y1 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx == 0:  # straight up and down line
                return 90
            line_angle = math.atan(dy / dx) * (180 / math.pi)
            if x1 > x2:
                line_angle = 180 - line_angle
            return line_angle

        def best_relative_angles(lines_one, lines_two, angle):
            """
                Find the line in each set that has the closest relative angle of angle to a line of the other set.
                Note: the neck line and nose line have a 30 degree difference
            """
            best_first = None
            best_second = None
            smallest_diff = float('inf')
            for first in lines_one:
                first_angle = find_line_angle(first)
                for second in lines_two:
                    second_angle = find_line_angle(second)
                    if abs(first_angle - second_angle - angle) < smallest_diff:
                        smallest_diff = abs(first_angle - second_angle - angle)
                        best_first = first
                        best_second = second
            return best_first, best_second

        def closest_angle_line(lines, angle):
            """
                Find the line in lines closest to the angle angle to get the most likely accurate detected line in the
                coin. For neck, use 90. For nose, use 60.
            """
            if lines is None:
                return None, None
            smallest_diff = None
            best_line = None
            best_line_angle = None
            for line in lines:
                line_angle = find_line_angle(line)
                if smallest_diff is None or abs(angle - line_angle) < smallest_diff:
                    best_line = line
                    smallest_diff = abs(angle - line_angle)
                    best_line_angle = line_angle
            return best_line, best_line_angle

        # Now, figure out some combination of all these to get the right angle of the coin 100% of the time!!!!!!!

        # if neckLines is None or noseLines is None:
        # # use the best angle line for each feature
        # best_nose, nose_angle = closest_angle_line(noseLines, 60)
        # best_neck, neck_angle = closest_angle_line(neckLines, 90)
        # use the longest line for each feature
        best_nose = find_longest(noseLines)
        best_neck = find_longest(neckLines)
        nose_angle = find_line_angle(best_nose)
        neck_angle = find_line_angle(best_neck)
        # else:
        #     # use the closest relative angle lines
        #     best_neck, best_nose = best_relative_angles(neckLines, noseLines, 30)
        #     display(c, best_neck, best_nose)
        #     neck_angle = findLineAngle(best_neck)
        #     nose_angle = findLineAngle(best_nose)
        print("NECK", neck_angle, "NOSE", nose_angle)

        goodNeckAngle = False
        goodNoseAngle = False

        # If the angle is close enough, don't bother rotating
        if nose_angle is not None and 58 < nose_angle < 62:
            goodNoseAngle = True
            # print("good nose angle", noseAngle)
        if neck_angle is not None and 89.5 < neck_angle < 91.5:
            goodNeckAngle = True
            # print("good neck angle", neckAngle)

        angle = 0
        if goodNoseAngle is False and goodNeckAngle is False:
            if neck_angle is not None and abs(90 - neck_angle) < 10:
                angle = 90 - neck_angle
            elif nose_angle is not None and abs(60 - nose_angle) < 10:
                angle = 60 - nose_angle
        self.coinangle = angle
        return angle

    def applymask(self, mask):  # computes features within a range determined by a mask image
        smoothed = cv2.GaussianBlur(self.grayimg, ksize=(0,0), sigmaX=1.5, sigmaY=1.5)
        (dxin, dyin) = cv2.spatialGradient(smoothed)
        dx = dxin.astype(np.float)
        dy = dyin.astype(np.float)
        # compute the grad first, then apply mask, so we don't have to deal with bondary issues
        gradmag = np.sqrt(np.add(np.multiply(dx, dx), np.multiply(dy, dy))).astype(np.int16)
        gradvec = gradmag.ravel()
        maskvec = mask.M.ravel()
        selection = (maskvec > 0)
        graddata = gradvec[selection]
        dxvec = np.abs(dx).ravel()
        dxdata = dxvec[selection]
        dyvec = np.abs(dy).ravel()
        dydata = dyvec[selection]

        meangrad = np.mean(graddata)
        mediangrad = np.median(graddata)
        mingrad = np.min(graddata)
        maxgrad = np.max(graddata)
        meandx = np.mean(dxdata)
        mediandx = np.median(dxdata)
        mindx = np.min(dxdata)
        maxdx = np.max(dxdata)
        meandy = np.mean(dydata)
        mediandy = np.median(dydata)
        mindy = np.min(dydata)
        maxdy = np.max(dydata)

        return (meangrad, mediangrad, mingrad, maxgrad,
                meandx, mediandx, mindx, maxdx,
                meandy, mediandy, mindy, maxdy)

    def featurenames(self, prefix):
        names = [prefix + '_meangrad',
                 prefix + '_mediangrad',
                 prefix + '_mingrad',
                 prefix + '_maxgrad',
                 prefix + '_meandx',
                 prefix + '_mediandx',
                 prefix + '_mindx',
                 prefix + '_maxdx',
                 prefix + '_meandy',
                 prefix + '_mediandy',
                 prefix + '_mindy',
                 prefix + '_maxdy']
        return names
