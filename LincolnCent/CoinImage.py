# Coins2022/CoinImage.py     Creed Jones     VT ECE  Sept 8, 2022
# development in support of the MDE coin grading teams AY22-23
# CoinImage objects are loaded, operated on, displayed and stored as needed

import cv2
import math
import numpy as np

class CoinImage():
    def __init__(self):
        self.colorimg = None
        self.grayimg = None
        self.binaryimg = None
        self.coincenter = None
        self.coinradius = None

    def load(self, fname):
        try:
            self.colorimg = cv2.imread(fname)
            self.grayimg = cv2.cvtColor(self.colorimg, cv2.COLOR_BGR2GRAY)
        except:
            self.colorimg = None
            self.grayimg = None
        return self.colorimg

    def save(self, fname):
        cv2.imwrite(fname, self.colorimg)

    def findcenter(self):
        blur = cv2.GaussianBlur(self.grayimg, ksize=(0, 0), sigmaX=1.5, sigmaY=1.5)
        thr, self.binaryimg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        blobs = detector.detect(self.binaryimg)

        mom = cv2.moments(self.binaryimg)
        self.coincenter = ( (mom["m10"] / mom["m00"]), (mom["m01"] / mom["m00"]) )
        self.coinradius = math.sqrt(mom["m00"]/(255*math.pi))
        pass

    def applymask(self, mask):
        smoothed = cv2.GaussianBlur(self.grayimg, ksize=(0,0), sigmaX=1.5, sigmaY=1.5)
        
        dxin = cv2.Sobel(smoothed, cv2.CV_16S, 1, 0, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        dyin = cv2.Sobel(smoothed, cv2.CV_16S, 0, 1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        #(dxin, dyin) = cv2.spatialGradient(smoothed)   # spatial gradient = 2x Sobel 3x3

        # Display edge-detected images

        #cv2.imshow('Smoothed Image', smoothed)
        #cv2.imshow('X Edges', dxin)
        #cv2.imshow('Y Edges', dyin)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        dx = dxin.astype(np.float)
        dy = dyin.astype(np.float)
        squared = np.add(np.multiply(dx, dx), np.multiply(dy, dy))
        gradmag = np.sqrt( np.add(np.multiply(dx, dx), np.multiply(dy, dy))).astype(np.int16)
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
        maxgrad = np.max(graddata)
        meandx = np.mean(dxdata)
        mediandx = np.median(dxdata)
        maxdx = np.max(dxdata)
        meandy = np.mean(dydata)
        mediandy = np.median(dydata)
        maxdy = np.max(dydata)
        return (meangrad, mediangrad, maxgrad,
                meandx, mediandx, maxdx,
                meandy, mediandy, maxdy)

        # old, includes min
        '''
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
        '''

    def featurenames(self, prefix):
        names = [ prefix + '_meangrad',
                  prefix + '_mediangrad',
                  prefix + '_maxgrad',
                  prefix + '_meandx',
                  prefix + '_mediandx',
                  prefix + '_maxdx',
                  prefix + '_meandy',
                  prefix + '_mediandy',
                  prefix + '_maxdy' ]
        return names

    # old, includes min    
    '''
    def featurenames(self, prefix):
        names = [ prefix + '_meangrad',
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
                  prefix + '_maxdy' ]
        return names
    '''