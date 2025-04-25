'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
'''

import numpy as np
import cv2
from scipy.signal import find_peaks, peak_prominences
import imutils


def gradeMorganByArea(image, area, name):

    #based on the area which is a string zoom in on an area and rotate the image
    #its bottom left, bottom right, upper left, upper right
    #name is the filename for graphs

    #all zooms are y THEN x 

    img = image
    #copy_img = image
    if area == "BL":
        img = imutils.rotate(img, angle=-12)
        

        """
        #Below will show an image with a box around the area
        #we are looking at, if left in it ruins the calc
        copy_img = imutils.rotate(image, angle=-12)
        top_left = (350,390)
        bottom_right = (400, 475)
        color = (0,0,255)
        thinkness = 2
        rect_img = cv2.rectangle(copy_img, top_left, bottom_right, color, thinkness)
        cv2.imshow('rect_img', rect_img)
        cv2.waitKey(0)
        """

        img = img[390:475, 350:400]

        mask1 = cv2.threshold(img, 256, 255, cv2.THRESH_BINARY)[1] #TRYING 250 instead of 200
        img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)

    elif area == "BR":
        img = imutils.rotate(img, angle=30)

        """
        #Below will show an image with a box around the area
        #we are looking at, if left in it ruins the calc
        copy_img = imutils.rotate(image, angle=30)
        top_left = (600,390)
        bottom_right = (650, 475)
        color = (0,0,255)
        thinkness = 2
        rect_img = cv2.rectangle(copy_img, top_left, bottom_right, color, thinkness)
        cv2.imshow('rect_img', rect_img)
        cv2.waitKey(0)
        """

        img = img[390:475,600:650]
        mask1 = cv2.threshold(img, 256, 255, cv2.THRESH_BINARY)[1]
        img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)


    elif area == "UL":
        img = imutils.rotate(img, angle=35)
        

        """
        #Below will show an image with a box around the area
        #we are looking at, if left in it ruins the calc
        copy_img = imutils.rotate(image, angle=35)
        top_left = (180,425)
        bottom_right = (270, 475)
        color = (0,0,255)
        thinkness = 2
        rect_img = cv2.rectangle(copy_img, top_left, bottom_right, color, thinkness)
        cv2.imshow('rect_img', rect_img)
        cv2.waitKey(0)
        """

        img = img[425:475, 180:270]
        mask1 = cv2.threshold(img, 256, 255, cv2.THRESH_BINARY)[1]
        img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)
        #img = img[425:475, 180:280]
        #zoom here
    elif area == "UR":
        img = imutils.rotate(img, angle=-43)

        """
        #Below will show an image with a box around the area
        #we are looking at, if left in it ruins the calc
        copy_img = imutils.rotate(image, angle=-43)
        top_left = (780,450)
        bottom_right = (880, 505)
        color = (0,0,255)
        thinkness = 2
        rect_img = cv2.rectangle(copy_img, top_left, bottom_right, color, thinkness)
        cv2.imshow('rect_img', rect_img)
        cv2.waitKey(0)
        """

        img = img[450:505,780:880]
        mask1 = cv2.threshold(img, 256, 255, cv2.THRESH_BINARY)[1]
        img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)
        #img = img[475:525, 800:850]
        #zoom here
    else:
        print("Error Invalid Area of Coin")

    
    f = np.fft.fft(img, n=None, axis =- 0) #This is a vertical transform
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    #print(magnitude_spectrum[0][0])
 
    summed_spec = magnitude_spectrum.sum(axis=1) #changed to 1
    #Now i need to average them
    count = len(magnitude_spectrum) #gets how many lists there are, or how many items were summed
    average_spec = []
    for value in summed_spec:
        average_spec.append(value/count)

    #print(average_spec)
    """
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image: ' + 'MSD 67'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum: ' + name), plt.xticks([]), plt.yticks([])
    plt.show()
    """

    #Now doing to try and find the maxima's

    #FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    #maxima = average_spec[60:130] # - HERE IS THE ZOOM
    #maxima = average_spec[40:90]
    maxima = average_spec
    maxima = np.array(maxima)
    spike_index = max(average_spec)

    #maxima = np.array(average_spec)
    #peaks, _ = find_peaks(maxima, height=(max(average_spec)-110))
    peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
    #peaks, properties = find_peaks(maxima, height=(max(average_spec)-110))
    prominences = peak_prominences(maxima, peaks)[0]
    #print(peaks)
    #properties["prominences"].max()
    #peaks = [9,13,24]
    #peaks = [8,9,10,12,13,14,23,24,25]

    #print(peaks) #Grabs X values of peaks
    """
    #plt.plot(maxima)
    #plt.plot(peaks, maxima[peaks], "x")
    #plt.title('MSD 67' + ' Harmonic Maximas Isolated')
    #plt.xlabel("Pixel Space")
    #plt.ylabel("Avg Magnitude per Pixel")
    #plt.show()
    """
    #Now we need to do some manipulation of the peaks and maxima section
    #Peaks -> array of found peaks
    #Maxima -> array of maxima values, maxima[peak] gets peak values

    ignore = False
    feature_rate_sum = 0
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
            feature_rate_sum += maxima_val - overall_min

    return feature_rate_sum
    #print("File: " + name + " | Feature Rating: " + str(feature_rate_sum))
    
    #return 0.123456

    
def gradeMorganSilverDollar(name):

    #name = '61.jpg'
    #name = input("Input Image Name: ")#
    
    orig_img = cv2.imread(name,0)
    #img = img[400:500, 325:375]

    

    Bleft_wing_val = gradeMorganByArea(orig_img, "BL", name)
    #print("File: " + name + " | Feature Rating BL: " + str(Bleft_wing_val))
    print("File: " + name + " | Feature Rating BL: " + str(round(Bleft_wing_val, 3)))

    BRight_wing_val = gradeMorganByArea(orig_img, "BR", name)
    print("File: " + name + " | Feature Rating BR: " + str(round(BRight_wing_val, 3)))

    Uleft_wing_val = gradeMorganByArea(orig_img, "UL", name)
    print("File: " + name + " | Feature Rating UL: " + str(round(Uleft_wing_val, 3)))

    Uright_wing_val = gradeMorganByArea(orig_img, "UR", name)
    print("File: " + name + " | Feature Rating UR: " + str(round(Uright_wing_val, 3)))

    #TESTING FLAGS HERE TO ACCOUNT FOR WING PROBLEMS
    wing_val_list = [] #this list contains no zeros
    errors_detected = 0 #this shouldnt ever go above 2, if its above two start throwing them out

    #check all wing vales to see if its zero, if its not then append it
    if Bleft_wing_val > 0:
        wing_val_list.append(Bleft_wing_val)
        Bleft_wing_val *= 1 #experimental trial to reward peak isolation
    if BRight_wing_val > 0:
        wing_val_list.append(BRight_wing_val)
        BRight_wing_val *= 1
    if Uleft_wing_val > 0:
        wing_val_list.append(Uleft_wing_val)
        Uleft_wing_val *= 1
    if Uright_wing_val > 0:
        wing_val_list.append(Uright_wing_val)
        Uright_wing_val *= 1

    max_wing_val = (min(wing_val_list))*1.0 #this name is wrong ill change it later

    if Bleft_wing_val == 0 and errors_detected < 2:
        Bleft_wing_val = max_wing_val
        errors_detected += 1
    if BRight_wing_val == 0 and errors_detected < 2:
        BRight_wing_val = max_wing_val
        errors_detected += 1
    if Uleft_wing_val == 0 and errors_detected < 2:
        Uleft_wing_val = max_wing_val
        errors_detected += 1
    if Uright_wing_val == 0 and errors_detected < 2:
        Uright_wing_val = max_wing_val
        errors_detected += 1

    avg_rating = (Bleft_wing_val+BRight_wing_val+Uleft_wing_val+Uright_wing_val)/4
    #print("File: " + name + " | Overall Feature Rating: " + str(round(avg_rating, 3)))

    trend = [0.019, -0.963, 14.441]
    x = avg_rating #for simplicity sake
    sheldon_scale_grade = x*x*trend[0] + x*trend[1] + trend[2]
    print("ESS: " + str(round(sheldon_scale_grade,3)))

    return([Bleft_wing_val, BRight_wing_val, Uleft_wing_val, Uright_wing_val, sheldon_scale_grade])
    