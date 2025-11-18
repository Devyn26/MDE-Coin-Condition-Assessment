'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
'''
from functools import lru_cache
from .ImageOpener import loadImages
from .ImageAdjuster import gaussian
import numpy as np
import matplotlib.pyplot as plt
import cv2
from . import ImageHSV
import time

def _now():
    return time.perf_counter()


def patternMatch(img, template, inpaint): # MORLEY

    if inpaint:
        mask1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
        img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)

        mask2 = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY)[1]
        template = cv2.inpaint(template, mask2, 0.1, cv2.INPAINT_TELEA)

    # Ensure both are valid
    if img is None or template is None:
        raise ValueError("patternMatch: One of the images is None.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if template.ndim == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Convert to same dtype
    img = img.astype(np.float32)
    template = template.astype(np.float32)

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # plt.imshow(cv2.normalize(res, None))
    # plt.show()
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    return res, maxVal, maxLoc

@lru_cache(maxsize=None)
def _get_templates_blurred(dirpath: str):
    t0 = _now()
    print(f"[DEBUG] [Templates] Loading from: {dirpath}")
    tpls = loadImages('grey', dirpath)
    t1 = _now()
    print(f"[DEBUG] [Templates] Loaded {len(tpls)} template(s) in {t1 - t0:.3f}s")

    tb0 = _now()
    tpls_blurred = tuple(gaussian(t) for t in tpls)
    tb1 = _now()
    print(f"[DEBUG] [Templates] Blurred {len(tpls_blurred)} template(s) in {tb1 - tb0:.3f}s")
    print(f"[DEBUG] [Templates] Total cache build time: {tb1 - t0:.3f}s")
    return tpls_blurred



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

def getCorrelation(image, templates):
    """
    Compute avg(max NCC) over a bank of templates.
    Assumes `templates` are already pre-blurred via _get_templates_blurred.
    """
    total0 = _now()


    b0 = _now()
    imageBlur = gaussian(image)
    b1 = _now()
    print(f"[DEBUG] [Match] Input blur: {(b1 - b0):.3f}s | image shape={image.shape}")

    maxVals = []
    m0 = _now()
    for idx, tBlur in enumerate(templates):
        tH, tW = tBlur.shape[:2]
        s0 = _now()
        res, maxVal, maxLoc = patternMatch(imageBlur, tBlur, False)  
        s1 = _now()
        print(f"[DEBUG] [Match] Template {idx+1}/{len(templates)} "
              f"size={tW}x{tH} | matchTemplate: {s1 - s0:.3f}s | max={maxVal:.5f} @ {maxLoc}")
        maxVals.append(maxVal)
    m1 = _now()

    feature = (sum(maxVals) / len(maxVals)) if maxVals else float('nan')
    total1 = _now()
    print(f"[DEBUG] [Match] Per-template loop: {(m1 - m0):.3f}s | feature(avg-max)={feature:.6f}")
    print(f"[DEBUG] [Match] Total getCorrelation: {(total1 - total0):.3f}s")

    return feature




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

import time
def _now(): return time.perf_counter()

def gradeCoin(path, isMSD, isBrown):
    g0 = _now()
    print(f"[DEBUG] [Grade] Start gradeCoin | path={path} | isMSD={isMSD} | isBrown={isBrown}")

    r0 = _now()
    img = path
    r1 = _now()
    print(f"[DEBUG] [Grade] Read image: {r1 - r0:.3f}s | shape={None if img is None else img.shape}")
    if img is None:
        raise IOError(f"Failed to read image for grading: {path}")

    # Get template
    tdir = './LincolnCent/Images/MSDTemplates/' if isMSD else './LincolnCent/Images/PatternMatchTemplate/'
    t0 = _now()
    templates = _get_templates_blurred(tdir)  
    t1 = _now()
    print(f"[DEBUG] [Grade] Templates ready in {t1 - t0:.3f}s | count={len(templates)}")


    c0 = _now()
    coefficients = []
    if isMSD:
        poly_file = './LincolnCent/MSD_Prediction_Function.txt'
    elif isBrown:
        poly_file = './LincolnCent/LHCB_Prediction_Function.txt'
    else:
        poly_file = './LincolnCent/LHCR_Prediction_Function.txt'

    with open(poly_file, 'r') as reader:
        for line in reader:
            coefficients.append(float(line))
    c1 = _now()
    print(f"[DEBUG] [Grade] Coeffs loaded from {poly_file} in {c1 - c0:.3f}s | count={len(coefficients)}")

    #Correlation feature 
    f0 = _now()
    correlation = getCorrelation(img, templates)   
    f1 = _now()
    print(f"[DEBUG] [Grade] getCorrelation: {f1 - f0:.3f}s | feature={correlation:.6f}")

    # polynomial eval
    p0 = _now()
    p = np.poly1d(coefficients)
    grade = p(correlation)
    p1 = _now()
    print(f"[DEBUG] [Grade] Poly eval: {p1 - p0:.6f}s | grade={float(grade):.3f}")

    g1 = _now()
    print(f"[DEBUG] [Grade] Total gradeCoin: {g1 - g0:.3f}s")
    return grade

if __name__ == '__main__':
    #generatePredictionFunctions()

    grade = gradeCoin("C:/Users/C_Fri/Desktop/SeniorDesign/CoinCherrypicker/Images/MSDs/1901MSD58.jpg", True, False)
    print("Predicted Sheldon Scale grade:", round(grade))
    print("Actual Sheldon Scale grade: 58")
    print("Feature definition (0 - 10):", round(grade / 70 * 10))