"""Deprecated"""
# import os
# import cv2
# import numpy as np
# from scipy import signal
#
# MAX_FEATURES = 10000
# MATCH_THRESHOLD = 0.15
#
#
# def alignImage(grayimg):
#
#     dirname = os.path.abspath('starter_kit_mostly_fixed/obverse/Morgan 1884-CC MS67 PCGS obverse.jpg')
#     template = cv2.imread(dirname)
#     template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#
#     orb = cv2.ORB_create(MAX_FEATURES)
#     keypoints, descriptors = orb.detectAndCompute(grayimg, None)
#     keypoints_t, descriptors_t = orb.detectAndCompute(template_gray, None)
#
#     matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#     matches = matcher.match(descriptors, descriptors_t, None)
#     # print(matches)
#     matches = sorted(matches, key=lambda x: x.distance, reverse=False)
#
#     numGoodMatches = int(len(matches) * MATCH_THRESHOLD)
#     matches = matches[:numGoodMatches]
#
#     imMatches = cv2.drawMatches(grayimg, keypoints, template_gray, keypoints_t, matches, None)
#     cv2.imshow("matches", imMatches)
#     cv2.waitKey(0)
#
#     pnts1 = np.zeros((len(matches), 2), dtype=np.float32)
#     pnts2 = np.zeros((len(matches), 2), dtype=np.float32)
#
#     for i, match in enumerate(matches):
#         pnts1[i, :] = keypoints[match.queryIdx].pt
#         pnts2[i, :] = keypoints_t[match.trainIdx].pt
#
#     h, mask = cv2.findHomography(pnts1, pnts2, cv2.RANSAC)
#
#     height, width, channels = template.shape
#     coinReg = cv2.warpPerspective(grayimg, h, (width, height))
#
#     return coinReg, h
