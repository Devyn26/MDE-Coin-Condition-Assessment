"""
Adjusts and saves ALL images in a directory to be the same
rotation, position, and scale so specific coin regions can be easily found
Will overwrite the original image

Author: Lizzie LaVallee
Date: 13 Feb 2023
"""
import numpy as np
import os
import cv2
from CoinImage import CoinImage
import EdgeEval
from matplotlib import pyplot as plt


def rotateImg(img, rotate_angle, center=(500, 500)):
    """ rotates rotate_angle degrees cc around center point """
    rot_mat = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, (1000, 1000), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return result


def rescaleImg(img, scale, center):
    """
        resizes an image (zoom in/out)
        scale < 1 for zoom out, > 1 for zoom in
    """
    rot_mat = cv2.getRotationMatrix2D(center, 0, scale)
    result = cv2.warpAffine(img, rot_mat, (1000, 1000), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result


def shiftImage(img, r_shift, d_shift):
    """ Shifts img r_shift to the right and d_shift down """
    translation_matrix = np.float32([[1, 0, r_shift], [0, 1, d_shift]])
    result = cv2.warpAffine(img, translation_matrix, (1000, 1000), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result


def findParameters(coin):
    """ Find the angle, center, and radius of the coin image """
    if coin.coincenter is None:
        coin.findcenter(None)
    angle = coin.findangle()
    return angle, coin.coincenter, coin.coinradius


def adjustImage(coin, targetAngle, targetCenter, targetRadius):
    """ Adjust this image to match the template dimensions """

    # # rotate the coin
    # rotateAngle = targetAngle - currAngle
    # self.stdimg = CoinImage.rotateImg(self.stdimg, rotateAngle, self.coincenter)


    # shift position
    right_shift = targetCenter[0] - coin.coincenter[0]
    down_shift = targetCenter[1] - coin.coincenter[1]
    if right_shift >= 1 or down_shift >= 1:
        coin.colorimg = shiftImage(coin.colorimg, right_shift, down_shift)

    # resize
    scale = targetRadius / coin.coinradius
    coin.colorimg = rescaleImg(coin.colorimg, scale, coin.coincenter)

    # reload the image. may want to change later
    return coin.colorimg


def adjustAndSaveImage(fname):
    coin = CoinImage()
    try:
        coin.load(fname)
    # rescale the image to be smaller and try again.
    # sometimes coins that are too large throws off the center/radius reading
    except:
        img = cv2.imread(fname)
        scaled = rescaleImg(img, .97, (500, 500))
        cv2.imwrite(fname[:-4] + "scaled.jpg", scaled)
        try:
            coin.load(fname[:-4] + "scaled.jpg")
            os.remove(fname[:-4] + "scaled.jpg")
        except:
            os.remove(fname[:-4] + "scaled.jpg")
            return False

    adjustImage(coin, 0, (500, 500), 485)

    # Make sure the adjusted image can be read as well
    coin.grayimg = cv2.cvtColor(coin.colorimg, cv2.COLOR_BGR2GRAY)
    try:
        coin.findcenter()
        coin.save(fname)
        return True
    except:
        return False


def fixRotation(filename):
    """
        TEMP FUNCTION to adjust the rotation of the coin, integrate with other adjustment function later
    """
    c = CoinImage()
    c.load(filename)
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.imshow("original", c.colorimg)
    cv2.waitKey(0)
    #angle = angleSilly(c)
    angle = c.findangle()
    result = rotateImg(c.colorimg, angle, c.coincenter)
    cv2.namedWindow("rotated", cv2.WINDOW_NORMAL)
    cv2.imshow('rotated', result)
    cv2.waitKey(0)


# def angleSilly(coin):
#     equalized = EdgeEval.histEqualization(coin.grayimg)
#     #blurred = cv2.bilateralFilter(src=equalized, d=5, sigmaColor=200, sigmaSpace=200)
#     #coin.grayimg = coin.grayimg[:500, :500]
#     #coin.colorimg = coin.colorimg[:500, :500]
#     blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
#     rawCanny = EdgeEval.cannyEdge(blurred, 50, 150)
#
#     ret, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
#
#     contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     image_copy = coin.grayimg.copy()
#     cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
#                      lineType=cv2.LINE_AA)
#
#     cv2.imshow('approximation', equalized)
#     cv2.waitKey(0)
#
#     # fig, ax = plt.subplots(1, 2)
#     # ax[0].imshow(rawCanny)
#     # ax[1].imshow(drawn_img)
#     # plt.show()
#
#     # cv2.imshow('Hough', lines_edges)
#     # cv2.waitKey(0)
#     return 0


if __name__ == '__main__':
    dir_name = os.path.abspath('ScrapedImages/obverse') + '\\'  # put the directory here
    fileList = os.listdir(dir_name)
    for index, filename in enumerate(fileList):
        # fixRotation(dir_name + filename)
        if adjustAndSaveImage(dir_name, filename) is False:
            print("Could not adjust", filename)


    # f = open("CouldntAdjust.txt", 'a')
    # dirname = os.path.abspath('ScrapedImages/obverse') + '\\'  # put the directory here
    # adjusteddir = os.path.abspath('ScrapedImages/obverse') + '\\'
    # # dirname = os.path.abspath('../../reverse_scaled') + '\\'  # put the directory here
    # fileList = os.listdir(dirname)
    # for index, filename in enumerate(fileList):
    #     c = CoinImage()
    #     c.load(dirname + filename)
    #     center_before = c.coincenter
    #     radius_before = c.coinradius
    #     try:
    #     # print("Coin Angle", )
    #     #     cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    #     #     cv2.imshow('original', c.colorimg)
    #     #     cv2.waitKey(0)
    #         adjusted = adjustImage(c, 0, (500, 500), 485)
    #     except Exception as e:
    #         print(e)
    #         f.write(filename + '\n')
    #         print("Couldn't adjust ****", filename, "****", index)
    #         continue
    #     c.save(adjusteddir + filename)
    #     c.grayimg = cv2.cvtColor(c.colorimg, cv2.COLOR_BGR2GRAY)
    #     try:
    #         c.findcenter()
    #     except:
    #         print("Couldn't find center of adjusted.", index)
    #         f.write(filename + '\n')
    #         continue
    #     # print("Center Change", center_before, c.coincenter, "Radius Change", radius_before, c.coinradius)
    #
    #     # cv2.namedWindow("adjusted", cv2.WINDOW_NORMAL)
    #     # cv2.imshow('adjusted', adjusted)
    #     # cv2.waitKey(0)
    # f.flush()
    # f.close()
