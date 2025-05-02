# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import os
#
# from CoinImage import CoinImage
# from MaskImage import MaskImage
#
#
# def findEdge(coinimg, wedges):
#     M = MaskImage()
#     quadDen = np.zeros(4)
#     for wedgect in range(len(wedges)):
#         M.clear()
#         M.create_wedge(start_angle=wedges[wedgect][0], end_angle=wedges[wedgect][1], circleRadius=900,
#                        wedgeRadius=3000, xc=int(coinimg.coincenter[0]), yc=int(coinimg.coincenter[1]))
#
#         inputImg = coinimg.grayimg
#         kernel = 7
#         sigma = 1.5
#
#         blurred = cv2.GaussianBlur(inputImg, ksize=(kernel, kernel), sigmaX=sigma, sigmaY=sigma)
#
#         edges = cv2.Canny(blurred, 40, 120)
#
#         maskedEdges = cv2.bitwise_and(M.M, edges)
#         #
#         # cv2.imshow('test', maskedEdges)
#         # cv2.waitKey(0)
#
#         white_pixels = np.sum(maskedEdges == 255)
#
#         quadDen[wedgect] = white_pixels
#
#     return quadDen
#
#
# def calculateEqualQuadrants(quadDens, denAngles, balDen, I, wedges):
#
#     iteration = 0.01
#     forgiveness = 20
#     # As of now currently the best I've managed to get for the obverse with out edges
#     # i = 0.01, f = 30
#     # ((0.48, 69.65), (69.65, 154.33), (154.33, 295.67), (295.67, 0.48))
#
#     # Without circle obverse:
#     # i = 0.01, f = 10
#     # ((2.34, 76.67), (76.67, 160.81), (160.81, 289.34), (289.34, 2.34))
#
#     # Without circle reverse:
#     # i = 0.01, f = 20
#     # ((0.01, 91.05), (91.05, 182.83), (182.83, 267.01), (267.01, 0.01))
#
#     cont = True
#     while cont:
#         cont = False
#         # Check if the top right is close to the ideal density
#         if abs(balDen - denAngles[0][0]) > forgiveness:
#             # Top right has less features than top left, so top right takes from top left
#             if denAngles[0][0] < denAngles[3][0] and denAngles[3][2] is False:
#                 cont = True
#                 denAngles[0][1][0] -= iteration
#                 # Adjusts angle to stay within the range of 0-360
#                 if denAngles[0][1][0] < 0:
#                     denAngles[0][1][0] += 360
#                 elif denAngles[0][1][0] > 360:
#                     denAngles[0][1][0] = iteration
#
#                 denAngles[3][1][1] -= iteration
#                 if denAngles[3][1][1] < 0:
#                     denAngles[3][1][1] += 360
#                 elif denAngles[3][1][1] > 360:
#                     denAngles[3][1][1] = iteration
#
#             # Top right has less features than bottom right, so top right takes from bottom right
#             if denAngles[0][0] < denAngles[1][0] and denAngles[1][2] is False:
#                 cont = True
#                 denAngles[0][1][1] += iteration
#                 denAngles[1][1][0] += iteration
#
#         else:
#             denAngles[0][2] = True
#
#         if abs(balDen - denAngles[1][0]) > forgiveness:
#             # Bottom right has less features than top right, so bottom right takes from top right
#             if denAngles[1][0] < denAngles[0][0] and denAngles[0][2] is False:
#                 cont = True
#                 denAngles[1][1][0] -= iteration
#                 denAngles[0][1][1] -= iteration
#
#             if denAngles[1][0] < denAngles[2][0] and denAngles[2][2] is False:
#                 cont = True
#                 denAngles[1][1][1] += iteration
#                 denAngles[2][1][0] += iteration
#         else:
#             denAngles[1][2] = True
#
#         if abs(balDen - denAngles[2][0]) > forgiveness:
#             # Bottom right has less features than top right, so bottom right takes from top right
#             if denAngles[2][0] < denAngles[1][0] and denAngles[1][2] is False:
#                 cont = True
#                 denAngles[2][1][0] -= iteration
#                 denAngles[1][1][1] -= iteration
#
#             if denAngles[2][0] < denAngles[3][0] and denAngles[3][2] is False:
#                 cont = True
#                 denAngles[2][1][1] += iteration
#                 denAngles[3][1][0] += iteration
#         else:
#             denAngles[2][2] = True
#
#         if abs(balDen - denAngles[3][0]) > forgiveness:
#             # Bottom right has less features than top right, so bottom right takes from top right
#             if denAngles[3][0] < denAngles[2][0] and denAngles[2][2] is False:
#                 cont = True
#                 denAngles[3][1][0] -= iteration
#                 denAngles[2][1][1] -= iteration
#
#             if denAngles[3][0] < denAngles[0][0] and denAngles[0][2] is False:
#                 cont = True
#                 denAngles[3][1][1] += iteration
#                 if denAngles[3][1][1] < 0:
#                     denAngles[3][1][1] += 360
#                 elif denAngles[3][1][1] > 360:
#                     denAngles[3][1][1] = iteration
#
#                 denAngles[0][1][0] += iteration
#                 if denAngles[0][1][0] < 0:
#                     denAngles[0][1][0] += 360
#                 elif denAngles[0][1][0] > 360:
#                     denAngles[0][1][0] = iteration
#         else:
#             denAngles[3][2] = True
#
#         wedges = ((denAngles[0][1][0], denAngles[0][1][1]), (denAngles[1][1][0], denAngles[1][1][1]),
#                   (denAngles[2][1][0], denAngles[2][1][1]), (denAngles[3][1][0], denAngles[3][1][1]))
#
#         print(denAngles)
#
#         quadDens = findEdge(coinimg=I, wedges=wedges)
#
#         denAngles[0][0] = quadDens[0]
#         denAngles[1][0] = quadDens[1]
#         denAngles[2][0] = quadDens[2]
#         denAngles[3][0] = quadDens[3]
#
#     return quadDens, wedges
#
# def main():
#
#     path = os.path.abspath('starter_kit_mostly_fixed/perfecto/Morgan 2021-CC MS70 _ reverse.jpg')
#     wedges = ((0, 90), (90, 180), (180, 270), (270, 0))
#
#     I = CoinImage()
#     I.load(path)
#     I.findcenter(True)
#     quadDens = findEdge(coinimg=I, wedges=wedges)
#     rigthBal = (quadDens[0] + quadDens[1]) / 2
#     leftBal = (quadDens[2] + quadDens[3]) / 2
#     balDen = (rigthBal + leftBal) / 2
#     # [quadrant density, wedge angle, protected]
#     denAngles = [[quadDens[0], [0.0, 90.0], False], [quadDens[1], [90.0, 180.0], False],
#                  [quadDens[2], [180.0, 270.0], False], [quadDens[3], [270.0, 0.0], False]]
#
#     quadDens, wedges = calculateEqualQuadrants(quadDens, denAngles, balDen, I, wedges)
#
#     print(quadDens)
#     print(wedges)

# Warning, this takes 8 billion years to run
# if __name__ == '__main__':
#     main()
