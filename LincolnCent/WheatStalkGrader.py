'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
'''

import numpy as np
import cv2
from scipy.signal import find_peaks, peak_prominences
#from ImageAdjuster import houghCenters, coinFlattener

def houghCenters(img):
    gray = cv2.medianBlur(img, 5)
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30)
    circles = np.reshape(circles, [circles.shape[1], circles.shape[2]])

    return circles

def coinFlattener(img):
    circles = houghCenters(img)

    flatendImg = cv2.linearPolar(img, [circles[0][0], circles[0][1]], circles[0][2], 0)

    return flatendImg

def gradeWheatStalkPenny(image_name):

    """
    Step 1) Get Images Name
    Step 2) Straighten it
    Step 3) 1D FFT
    Step 4) Average
    Step 5) Peak analysis
    Step 6) Plug into equation

    OLD Extracted Slope: 0.47855280471223094
    OLD Extracted Intercept: 16.095850552588697
    Extracted Slope: 1.1507891082209682
    Extracted Intercept: 40.45697117990882
    """

    b = 40.45697117990882
    m = 1.1507891082209682

    #image_name = input("Enter the Image Name: ")

    #print("Image Name: " + image_name)

    original_img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

    #print("Img Data:")
    #print(img)
    
    flat = coinFlattener(original_img)

    #print("Flat Data:")
    #print(flat)

    #cv2.imshow('image', flat)
        
    #cv2.waitKey(0)

    img = flat[520:580, 720:850]

    #cv2.imshow("Flat", flat)
    #cv2.waitKey(0)

    #cv2.imshow("Cropped", img)
    #cv2.waitKey(0)

    mask1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)

    f = np.fft.fft(img, n=None, axis =- 1) #This is a horizontal transform
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    summed_spec = magnitude_spectrum.sum(axis=0)
    #Now i need to average them
    count = len(magnitude_spectrum) #gets how many lists there are, or how many items were summed
    average_spec = []
    for value in summed_spec:
        average_spec.append(value/count)

    #plt.subplot(121),plt.imshow(img, cmap = 'gray')
    #plt.title('Input Image:'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    #plt.title('Magnitude Spectrum:'), plt.xticks([]), plt.yticks([])
    #plt.show()  


    maxima = average_spec[40:90]
    maxima = np.array(maxima)
    spike_index = max(average_spec)

    
    peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
    prominences = peak_prominences(maxima, peaks)[0]

    
    #plt.plot(maxima)
    #plt.plot(peaks, maxima[peaks], "x")
    #plt.title('Harmonic Maximas Isolated')
    #plt.xlabel("Pixel Space")
    #plt.ylabel("Average Magnitude/Pixel")
    #plt.show()
    

    ignore = False
    feature_rate_sum_left = 0
    overall_min = min(maxima)

    for peak in peaks:
        #loop over all peaks
        if (len(peaks) == 1):
            print("No maxima located, low grade coin")
        #If the peak is the coherence speak ignore it
        if (maxima[peak] == max(maxima)):
            ignore = True
        elif(ignore == False):
            #do the stuff here
            maxima_val = maxima[peak] #gets the maxima value at that peak
            maxima_val_minus1 = maxima[peak-1]
            #feature_rate_sum += (maxima_val - maxima_val_minus1) #take difference between peak and next available point
            feature_rate_sum_left += maxima_val - overall_min


    """
    Second Area
    """
    img = flat[930:980, 720:850]

    #cv2.imshow("Cropped", img)
    #cv2.waitKey(0)

    mask1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)

    f = np.fft.fft(img, n=None, axis =- 1) #This is a horizontal transform
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    #print(magnitude_spectrum[0][0])
    
    summed_spec = magnitude_spectrum.sum(axis=0)
    #Now i need to average them
    count = len(magnitude_spectrum) #gets how many lists there are, or how many items were summed
    average_spec = []
    for value in summed_spec:
        average_spec.append(value/count)

    #print(average_spec) 
    

    #Now doing to try and find the maxima's

    #FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    maxima = average_spec[40:90]
    maxima = np.array(maxima)
    spike_index = max(average_spec)

    peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
    prominences = peak_prominences(maxima, peaks)[0]

    ignore = False
    feature_rate_sum_right = 0
    overall_min = min(maxima)

    for peak in peaks:
        #loop over all peaks
        if (len(peaks) == 1):
            print("No maxima located, low grade coin")
        #If the peak is the coherence speak ignore it
        if (maxima[peak] == max(maxima)):
            ignore = True
        elif(ignore == False):
            #do the stuff here
            maxima_val = maxima[peak] #gets the maxima value at that peak
            maxima_val_minus1 = maxima[peak-1]
            #feature_rate_sum += (maxima_val - maxima_val_minus1) #take difference between peak and next available point
            feature_rate_sum_right += maxima_val - overall_min


    #Now lastly, take the feature_rate_sum and plug into equation
    #in my equation y = mx + b, x is sheldon scale so
    # x = (y-b)/m
    print("Left Wheat Stalk Rating: " + str(round(feature_rate_sum_left,3)))
    print("Right Wheat Stalk Rating: " + str(round(feature_rate_sum_right,3)))

    feature_rate_sum_avg = (feature_rate_sum_left + feature_rate_sum_right)/2

    #sheldon_scale_grade = ((feature_rate_sum_avg - b)/m)
    trend = [0.00014162340303767857, -0.05219235501283228, 6.332637857746728, -187.17387181557189]
    x = feature_rate_sum_avg #for simplicity sake
    sheldon_scale_grade = x*x*x*trend[0] + x*x*trend[1] + x*trend[2] + trend[3]

    print("Estimated Wheat Stalk Sheldon Scale Grade: " + str(round(sheldon_scale_grade)))

    #cv2.imshow("Original Image", original_img)
    #cv2.waitKey(0)
    return([feature_rate_sum_left, feature_rate_sum_right, sheldon_scale_grade])


#values = gradeWheatStalkPenny("C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/Reverse/1909PennyBack30.jpg")
#print(values)







