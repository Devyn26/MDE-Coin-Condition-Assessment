'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
'''

from .ImageOpener import loadImages
from .ImageAdjuster import gaussian
import numpy as np
import matplotlib.pyplot as plt
import cv2
from . import ImageHSV

def patternMatch(img, template, inpaint):

    if inpaint:
        mask1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
        img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)

        mask2 = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY)[1]
        template = cv2.inpaint(template, mask2, 0.1, cv2.INPAINT_TELEA)

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # plt.imshow(cv2.normalize(res, None))
    # plt.show()
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    return res, maxVal, maxLoc

####################################DEPRICATED###########################################
def shapeFromShading(img, light):

    lightN = np.zeros(light.shape)
    for i in range(light.shape[0]):
        lightN[i,:] = light[i] / np.linalg.norm(light[i])

    b = np.ones([img.shape[1], img.shape[2], 3], np.double)
    p = np.ones(b.shape[:2], np.double)
    q = p
    Z = np.ones(p.shape,np.double)

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            E = np.transpose(img[:,i,j])

            tb = np.linalg.inv(np.transpose(lightN) @ lightN) @ np.transpose(lightN) @ E
            ntb = np.linalg.norm(tb)

            if ntb == 0:
                b[i,j,:] = 0
            else:
                b[i,j,:] = tb / ntb

            tM = b[i,j,:]
            ntb = np.linalg.norm(tM)

            if ntb == 0:
                tM = [0,0,0]
            else:
                tM = tM/ntb

            p[i,j] = tM[0]
            q[i,j] = tM[1]

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = (np.sum(q[:i,0]) + np.sum(p[i,:j]))

    return Z
##################################################################################################

def getCorrelation(image, templates):
    maxVals = []

    # plt.imshow(image, cmap=plt.get_cmap('gray'))
    # plt.show()

    imageBlur = gaussian(image)

    # plt.imshow(imageBlur, cmap=plt.get_cmap('gray'))
    # plt.show()
    
    for t in templates:
        #tAdjust = setSat(t, 50)
        templateBlur = gaussian(t)
        res, maxVal, maxLoc = patternMatch(imageBlur, templateBlur, False)
        maxVals.append(maxVal)
        
    return sum(maxVals) / len(maxVals)

def setSat(img, sat):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    s = sat * np.ones(s.shape, np.uint8)

    hsv = cv2.merge((h,s,v))
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

def generatePredictionFunctions():
    obverse = loadImages('grey', './LincolnCent/Images/MSDs/')
    templates = loadImages('grey', './LincolnCent/Images/MSDTemplates/')

    grades = np.array([62, 62, 3, 45, 61, 62, 63, 63, 4, 8, 58, 50, 50, 50, 50, 65, 4, 12, 53, 50, 20, 40, 45, 4, 12, 35, 68, 58, 35])

    average = []
    for img in obverse:
        average.append(getCorrelation(img, templates))

    MSDp = np.poly1d(np.polyfit(average, grades, 3))

    plt.figure(0)
    plt.plot(np.arange(0.4, 1, 0.05), MSDp(np.arange(0.4, 1, 0.05)))

    plt.plot(average, grades, 'ro')

    plt.grid()
    plt.title('Sheldon Scale as function of Confidence Value - MSD')
    plt.ylabel('Sheldon Scale Grade')
    plt.xlabel('Correlation Coefficient')
    
    error = np.sum([ (average[i] - MSDp(average[i])) ** 2 for i in range(len(average)) ])
    print("Model error:", error)

    plt.show()

    save = input("Save prediction function? (y/n) ")
    if save == "y":
        with open('./LincolnCent/MSD_Prediction_Function.txt', 'w') as writer:
            for coefficient in MSDp.coefficients:
                writer.write(str(coefficient) + '\n')

    obverse = loadImages('grey', './LincolnCent/Images/Obverse/Brown/')
    templates = loadImages('grey', './LincolnCent/Images/PatternMatchTemplate/')

    grades = np.array([8, 10, 12, 14, 20, 30, 35, 40, 45, 63, 64, 58, 12, 40])

    average = []
    for img in obverse:
        average.append(getCorrelation(img, templates))

    LHCBp = np.poly1d(np.polyfit(average, grades, 3))

    plt.figure(1)
    plt.plot(np.arange(0.5, 0.7, 0.05), LHCBp(np.arange(0.5, 0.7, 0.05)))

    plt.plot(average, grades, 'ro')

    plt.grid()
    plt.title('Sheldon Scale as function of Confidence Value - LHC (Brown)')
    plt.ylabel('Sheldon Scale Grade')
    plt.xlabel('Correlation Coefficient')
    
    error = np.sum([ (average[i] - LHCBp(average[i])) ** 2 for i in range(len(average)) ])
    print("Model error:", error)

    plt.show()

    save = input("Save prediction function? (y/n) ")
    if save == "y":
        with open('./LincolnCent/LHCB_Prediction_Function.txt', 'w') as writer:
            for coefficient in LHCBp.coefficients:
                writer.write(str(coefficient) + '\n')

    obverse = loadImages('grey', './LincolnCent/Images/Obverse/Red/')
    templates = loadImages('grey', './LincolnCent/Images/PatternMatchTemplate/')

    grades = np.array([63, 64, 65, 66, 67, 64, 65, 67, 66])

    average = []
    for img in obverse:
        average.append(getCorrelation(img, templates))

    LHCRp = np.poly1d(np.polyfit(average, grades, 3))

    plt.figure(2)
    plt.plot(np.arange(0.4, 0.6, 0.05), LHCRp(np.arange(0.4, 0.6, 0.05)))

    plt.plot(average, grades, 'ro')

    plt.grid()
    plt.title('Sheldon Scale as function of Confidence Value - LHC (Red)')
    plt.ylabel('Sheldon Scale Grade')
    plt.xlabel('Correlation Coefficient')
    
    error = np.sum([ (average[i] - LHCRp(average[i])) ** 2 for i in range(len(average)) ])
    print("Model error:", error)

    plt.show()

    save = input("Save prediction function? (y/n) ")
    if save == "y":
        with open('./LincolnCent/LHCR_Prediction_Function.txt', 'w') as writer:
            for coefficient in LHCRp.coefficients:
                writer.write(str(coefficient) + '\n')

def imgIsMSD(path):

    percentage = ImageHSV.Image_HSV_Region1(path)
    return False
    if percentage < 3:
        return True
    return False

def gradeCoin(path, isMSD, isBrown):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    coefficients = []
    
    if isMSD:
        templates = loadImages('grey', './LincolnCent/Images/MSDTemplates/')
        
        with open('./LincolnCent/MSD_Prediction_Function.txt', 'r') as reader:
            for line in reader:
                coefficients.append(float(line))
    elif isBrown:
        templates = loadImages('grey', './LincolnCent/Images/PatternMatchTemplate/')

        with open('./LincolnCent/LHCB_Prediction_Function.txt', 'r') as reader:
            for line in reader:
                coefficients.append(float(line))
    else:
        templates = loadImages('grey', './LincolnCent/Images/PatternMatchTemplate/')

        with open('./LincolnCent/LHCR_Prediction_Function.txt', 'r') as reader:
            for line in reader:
                coefficients.append(float(line))

    correlation = getCorrelation(img, templates)
    p = np.poly1d(coefficients)
    return p(correlation)

if __name__ == '__main__':
    #generatePredictionFunctions()

    grade = gradeCoin("C:/Users/C_Fri/Desktop/SeniorDesign/CoinCherrypicker/Images/MSDs/1901MSD58.jpg", True, False)
    print("Predicted Sheldon Scale grade:", round(grade))
    print("Actual Sheldon Scale grade: 58")
    print("Feature definition (0 - 10):", round(grade / 70 * 10))



    