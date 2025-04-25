# Coins2022/MaskImage.py     Creed Jones     VT ECE  Sept 8, 2022
# development in support of the MDE coin grading teams AY22-23
# MaskImages are binary - white (255) in the region of interest, black otherwise

import math
import numpy as np
import cv2

class MaskImage():
    def __init__(self, rows=1000, cols=1000):
        self.M = np.zeros( (rows, cols), dtype=np.uint8)
        self.points = None

    def clear(self):
        self.M.fill(0)

    def math_angle_to_nautical(self, ang):      # takes a math angle in radians and converts to nautical in degrees
        result = 90+(180.0/np.pi)*ang
        while (result > 360):
            result -= 360
        while (result < 0):
            result += 360
        return result

    def nautical_angle_to_math(self, ang):      # takes a nautical angle in degrees and converts to math, in radians
        result = (np.pi/180.0)*(-ang+90)
        while (result > np.pi):
            result -= 2*np.pi
        while (result < -np.pi):
            result += 2*np.pi
        return result

    def create_wedge(self, start_angle = 0, end_angle=45, radius=2000, xc=500, yc=500):
        start_math_angle = self.nautical_angle_to_math(start_angle)
        end_math_angle = self.nautical_angle_to_math(end_angle)
        start_angle_point = (int(radius*math.cos(start_math_angle)+xc), int(yc-radius*math.sin(start_math_angle)))
        end_angle_point = (int(radius*math.cos(end_math_angle)+xc), int(yc-radius*math.sin(end_math_angle)))

        self.points = np.asarray([(xc, yc), start_angle_point, end_angle_point, (xc, yc)])
        cv2.fillPoly(img=self.M, pts=[self.points], color=(255))

        # print('start_angle = {}, {}, end_angle = {}, {}'.
        #      format(start_angle, start_math_angle, end_angle, end_math_angle))
        # print("points are {}".format(self.points))


    def save(self, file):
        cv2.imwrite(file, self.M)

