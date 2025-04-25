"""
Code From:
https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Fourier_Transform_FFT_DFT.php

Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import multiprocessing
from ImageOpener import loadImages
from scipy.signal import find_peaks, peak_prominences
import imutils


def Transform_VG08_Wheat():
    img = cv2.imread('./CoinCherrypicker/Images/vg08_wheat_reverse.jpg',0)


    #img = img[175:400,0:100] #This isolates the whole wheat stalk
    img = img[175:235, 10:80] #This isolates just the part above the wheat stalk

    img_float32 = np.float32(img)

    #dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft = cv2.dft(img_float32, flags = cv2.DFT_ROWS+cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()    

def Transform_MS67_Wheat():
    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/ms67_wheat_reverse.jpg',0)

    #img = img[175:400,0:100]
    img = img[175:235, 10:80]

    img_float32 = np.float32(img)

    #dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft = cv2.dft(img_float32, flags = cv2.DFT_ROWS+cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()   

def Transform_MS67_Memorial():
    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/ms67_memorial_reverse.jpg',0)

    img = img[200:305,50:300]

    img = img[200:305,50:300]

    img = img[200:305,50:300]

    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()   


def OneD_Memorial_Transform():
    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/ms67_memorial_reverse.jpg',0)
    #img = img[200:305,50:300]
    img = img[270:300, 100:250]

    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags = cv2.DFT_ROWS+cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()   

def OneD_Wheat_Transform():
    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/straight.png',0)
    #img = img[200:305,50:300]
    #img = img[270:300, 100:250]

    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags = cv2.DFT_ROWS+cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()   

def Numpy_Memorial_Transform():

    """
    https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Signal_Processing_with_NumPy_Fourier_Transform_FFT_DFT_2.php
    """

    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/ms67_memorial_reverse.jpg',0)
    #img = img[200:305,50:300]
    img = img[270:300, 100:250]

    f = np.fft.fft(img, n=None, axis =- 0)

    f = np.fft.fft(img, n=None, axis =- 1) #This is a vertical transform
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    summed_spec = magnitude_spectrum.sum(axis=0)
    #Now i need to average them
    count = len(magnitude_spectrum) #gets how many lists there are, or how many items were summed
    average_spec = []
    for value in summed_spec:
        average_spec.append(value/count)

    #print(average_spec)
    """
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()  
    """
    plt.plot(average_spec)
    plt.title('Average Frequency - Memorial')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

def Template_Overlay():
    image1 = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/template1.jpg',cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/template2.jpg',cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/template3.jpg',cv2.IMREAD_GRAYSCALE)

    #cv2.imshow("Test comparison", image3)
    #cv2.waitKey(0)

    output1 = image1.copy()
    output2 = image1.copy()

    alpha = 0.5

    cv2.addWeighted(image1, alpha, image2, 1-alpha, 0, output1)
    #cv2.imshow("2 coins", output1)
    #cv2.waitKey(0)

    cv2.addWeighted(output1, alpha, image3, 1-alpha, 0, output2)
    output2 = output2[250:285,25:110]
    cv2.imshow("3 coins", output2)
    cv2.waitKey(0)
    cv2.imwrite('template_output.jpg', output2)

    #first overlay 1 and 2

def Numpy_VG08_Transform():

    """
    https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Signal_Processing_with_NumPy_Fourier_Transform_FFT_DFT_2.php
    """

    """ SMOOTHING TUTORIAL
    https://www.geeksforgeeks.org/how-to-plot-a-smooth-curve-in-matplotlib/
    """


    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/vg08_straight.png',0)
    #img = img[200:305,50:300]
    #img = img[350:600, 700:880]
    img = img[515:580, 700:880]

    f = np.fft.fft(img, n=None, axis =- 1)
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
    """
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()  
    """

    x = []
    i = 0
    length = len(average_spec)
    while i < length:
        x.append(i)
        i += 1


    #plt.plot(average_spec)
    plt.plot(average_spec)
    plt.title('VG08 Average Frequency - Wheat')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    #Maxima stuff here to

    maxima = average_spec[60:130]
    maxima = np.array(maxima)
    #maxima = np.array(average_spec)
    #peaks, _ = find_peaks(maxima, height=(max(average_spec)-110))
    peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
    #peaks, properties = find_peaks(maxima, height=(max(average_spec)-110))
    prominences = peak_prominences(maxima, peaks)[0]
    print(prominences)
    #properties["prominences"].max()

    #print(peaks) #Grabs X values of peaks

    plt.plot(maxima)
    plt.plot(peaks, maxima[peaks], "x")
    plt.title('VG08 Harmonic Maximas Isolated')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    #print(max(average_spec))

def Numpy_MS67_Transform():

    """
    https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Signal_Processing_with_NumPy_Fourier_Transform_FFT_DFT_2.php
    """

    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/ms67_straight.png',0)
    #img = img[325:575, 650:850]
    img = img[480:540, 650:840] #note this is y, then x

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
    """
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()  
    """
    plt.plot(average_spec)
    plt.title('MS67 Average Frequency - Wheat')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    #Now doing to try and find the maxima's

    #FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    maxima = average_spec[60:130]
    maxima = np.array(maxima)
    #maxima = np.array(average_spec)
    #peaks, _ = find_peaks(maxima, height=(max(average_spec)-110))
    peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
    #peaks, properties = find_peaks(maxima, height=(max(average_spec)-110))
    prominences = peak_prominences(maxima, peaks)[0]
    print(prominences)
    #properties["prominences"].max()

    plt.plot(maxima)
    plt.plot(peaks, maxima[peaks], "x")
    plt.title('MS67 Harmonic Maximas Isolated')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()
    #plt.plot(maxima)
    #plt.show()

    #print(max(average_spec))

def Numpy_MS40_Transform():

    """
    https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Signal_Processing_with_NumPy_Fourier_Transform_FFT_DFT_2.php
    """

    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/Reverse/straight/40.jpg',0)
    #img = img[325:575, 650:850]
    img = img[480:540, 650:840] #note this is y, then x

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
    """
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()  
    """
    plt.plot(average_spec)
    plt.title('MS40 Average Frequency - Wheat')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    #Now doing to try and find the maxima's

    #FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    maxima = average_spec[60:130]
    maxima = np.array(maxima)
    spike_index = max(average_spec)

    #maxima = np.array(average_spec)
    #peaks, _ = find_peaks(maxima, height=(max(average_spec)-110))
    peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
    #peaks, properties = find_peaks(maxima, height=(max(average_spec)-110))
    prominences = peak_prominences(maxima, peaks)[0]
    print(prominences)
    #properties["prominences"].max()

    #print(peaks) #Grabs X values of peaks

    plt.plot(maxima)
    plt.plot(peaks, maxima[peaks], "x")
    plt.title('MS40 Harmonic Maximas Isolated')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()
    #plt.plot(maxima)
    #plt.show()

    #print(max(average_spec))

def Overall_Fourier_to_Sheldon_Wheat():

    #Array used to hold X values - Known Sheldon scales
    sheldon_scales = [67, 8]

    #Array used to hold Y values - max's of the fourier graph
    freq_peaks = []

    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/ms67_straight.png',0)
    #img = img[325:575, 650:850]
    img = img[480:540, 650:840] #note this is y, then x

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

    freq_peaks.append(max(average_spec))

    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/vg08_straight.png',0)
    #img = img[325:575, 650:850]
    img = img[480:540, 650:840] #note this is y, then x

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

    freq_peaks.append(max(average_spec))

    print(freq_peaks)

    plt.plot(sheldon_scales, freq_peaks)
    plt.title('Peak Frequency vs. Sheldon Scale')
    plt.xlabel("Sheldon Scale")
    plt.ylabel("Peak Frequency")
    plt.show()


def Overall_Fourier_to_Sheldon_Memorial():

    #Array used to hold X values - Known Sheldon scales
    sheldon_scales = [63, 67, 69, 70]

    #Array used to hold Y values - max's of the fourier graph
    freq_peaks = []

    """
    Image 1
    """
    
    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/MS63_memorial_reverse.jpg',0)
    #img = img[325:575, 650:850]
    img = img[560:650, 300:700] #note this is y, then x

    #cv2.imshow("Zoom", img)
    #cv2.waitKey(0)

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

    freq_peaks.append(max(average_spec))

    plt.plot(average_spec)
    plt.title('MS63 Average Frequency - Memorial')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()
    
    """
    #Image 2
    """
    

    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/MS67_memorial_reverse.jpg',0)
    #img = img[325:575, 650:850]
    img = img[270:300, 100:250] #note this is y, then x

    #cv2.imshow("Zoom", img)
    #cv2.waitKey(0)

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

    freq_peaks.append(max(average_spec))

    plt.plot(average_spec)
    plt.title('MS67 Average Frequency - Memorial')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    """
    #Image 3
    """
    
    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/MS69_memorial_reverse.jpg',0)
    #img = img[325:575, 650:850]
    img = img[560:650, 300:700] #note this is y, then x

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

    freq_peaks.append(max(average_spec))

    plt.plot(average_spec)
    plt.title('MS69 Average Frequency - Memorial')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    """
    #Image 4
    """
    
    img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/MS70_memorial_reverse.jpg',0)
    #img = img[325:575, 650:850]
    img = img[270:305, 100:245] #note this is y, then x

    #cv2.imshow("Zoom", img)
    #cv2.waitKey(0)

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

    freq_peaks.append(max(average_spec))

    plt.plot(average_spec)
    plt.title('MS70 Average Frequency - Memorial')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    

    plt.plot(sheldon_scales, freq_peaks)
    plt.title('Peak Frequency vs. Sheldon Scale - Memorial')
    plt.xlabel("Sheldon Scale")
    plt.ylabel("Peak Frequency")
    plt.show()

def Giga_Opener_Wheat_Overall_Freq_Peak():

    images = loadImages('gray')
    freq_peaks = []
    #sheldon_scales = [45,8,10,15,20,30,35,64,20,58,12,25,40,58,63,65,68,66,67]
    sheldon_scales = [8,10,12,15,20,20,25,30,35,40,45,58,58,63,64,65,66,67,68]
    index = 0
    for img in images:
        #testing
        img = img[480:540, 650:840] #note this is y, then x

        f = np.fft.fft(img, n=None, axis =- 1) #This is a horizontal transform
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
 
        summed_spec = magnitude_spectrum.sum(axis=0)
        #Now i need to average them
        count = len(magnitude_spectrum) #gets how many lists there are, or how many items were summed
        average_spec = []
        for value in summed_spec:
            average_spec.append(value/count)

        freq_peaks.append(max(average_spec))
        index += 1
        print("Images Analyzed: " + str(index))

    print(freq_peaks)

    if (len(freq_peaks) is not len(sheldon_scales)):
        print("Error: Length's not the same")

    plt.plot(sheldon_scales, freq_peaks)
    plt.title('Peak Frequency vs. Sheldon Scale - Wheat')
    plt.xlabel("Sheldon Scale")
    plt.ylabel("Peak Frequency")
    plt.show()

def Given_Image_Name_Grade(name):
    
    orig_img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/Reverse/straight/' + name,0)
    #img = img[325:575, 650:850]
    #img = img[480:540, 650:840] #note this is y, then x
    #img = img[520:580, 690:880]
    img = orig_img[520:580, 720:850]
    #img = cv2.GaussianBlur(img,(5,5),sigmaX=0.75,sigmaY=0.75)

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
    
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image: ' + name), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum: ' + name), plt.xticks([]), plt.yticks([])
    plt.show()  
    

    #Now doing to try and find the maxima's

    #FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    #maxima = average_spec[60:130] # - HERE IS THE ZOOM
    maxima = average_spec[40:90]
    maxima = np.array(maxima)
    spike_index = max(average_spec)

    #maxima = np.array(average_spec)
    #peaks, _ = find_peaks(maxima, height=(max(average_spec)-110))
    peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
    #peaks, properties = find_peaks(maxima, height=(max(average_spec)-110))
    prominences = peak_prominences(maxima, peaks)[0]
    print(peaks)
    #properties["prominences"].max()
    #peaks = [9,13,24]
    #peaks = [8,9,10,12,13,14,23,24,25]

    #print(peaks) #Grabs X values of peaks

    plt.plot(maxima)
    plt.plot(peaks, maxima[peaks], "x")
    plt.title(name + ' Harmonic Maximas Isolated')
    plt.xlabel("Pixel Space")
    plt.ylabel("Avg Magnitude per Pixel")
    plt.show()

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

    print("File: " + name + " | Feature Rating Left Side: " + str(feature_rate_sum))



    """
    OTHER AREA HERE
    """
    img = orig_img[930:980, 720:850]
    #img = cv2.GaussianBlur(img,(5,5),sigmaX=0.75,sigmaY=0.75)

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
    
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image: ' + name), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum: ' + name), plt.xticks([]), plt.yticks([])
    plt.show()  
    

    #Now doing to try and find the maxima's

    #FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    #maxima = average_spec[60:130] # - HERE IS THE ZOOM
    maxima = average_spec[40:90]
    maxima = np.array(maxima)
    spike_index = max(average_spec)

    #maxima = np.array(average_spec)
    #peaks, _ = find_peaks(maxima, height=(max(average_spec)-110))
    peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
    #peaks, properties = find_peaks(maxima, height=(max(average_spec)-110))
    prominences = peak_prominences(maxima, peaks)[0]
    print(peaks)
    #properties["prominences"].max()
    #peaks = [9,13,24]
    #peaks = [8,9,10,12,13,14,23,24,25]

    #print(peaks) #Grabs X values of peaks

    plt.plot(maxima)
    plt.plot(peaks, maxima[peaks], "x")
    plt.title(name + ' Harmonic Maximas Isolated')
    plt.xlabel("Pixel Space")
    plt.ylabel("Avg Magnitude per Pixel")
    plt.show()

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

    print("File: " + name + " | Feature Rating Right Side: " + str(feature_rate_sum))


def Overall_Derivative_Plot():

    images = loadImages('gray') #WONT WORK ANYMORE CAUSE OPENER WAS CHANGED
    features_list = []
    #sheldon_scales = [45,8,10,15,20,30,35,64,20,58,12,25,40,58,63,65,68,66,67]
    #sheldon_scales = [8,10,12,15,20,20,25,30,35,40,45,58,58,63,64,65,66,67,68]
    sheldon_scales = [8,10,15,20,30,35,40,45,58,63,64,66,68] #removed 65 when souldnt have, and 20 and 35 for fitting
    #sheldon_scales = [8,10,15,30,40,45,58,63,64,66,68] #removed 10, 15, 45, 63 , 68, 30
    #sheldon_scales = [8,40,58,64,66] #final fitting points
    index = 0
    for orig_img in images:

        #img = img[520:580, 690:880]
        img = orig_img[520:580, 720:850]

        mask1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
        img = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)

        #img = cv2.GaussianBlur(img,(5,5),sigmaX=0.25,sigmaY=0.25)

        f = np.fft.fft(img, n=None, axis =- 1) #This is a horizontal transform
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

 
        summed_spec = magnitude_spectrum.sum(axis=0)
        #Now i need to average them
        count = len(magnitude_spectrum) #gets how many lists there are, or how many items were summed
        average_spec = []
        for value in summed_spec:
            average_spec.append(value/count)


        #Now doing to try and find the maxima's

        #FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

        #maxima = average_spec[60:130]
        maxima = average_spec[40:90]
        maxima = np.array(maxima)
        spike_index = max(average_spec)

    
        peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
        #peaks, properties = find_peaks(maxima, height=(max(average_spec)-110))
        prominences = peak_prominences(maxima, peaks)[0]
        print(peaks)


        #Now we need to do some manipulation of the peaks and maxima section
        #Peaks -> array of found peaks
        #Maxima -> array of maxima values, maxima[peak] gets peak values

        #peaks = [8,9,10,12,13,14,23,24,25] #added here for isolation harmonics - taken out so that i can return to better method

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

        #index += 1
        #feature_rate_sum_avg = (feature_rate_sum_left + feature_rate_sum_right) / 2
        #features_list.append(feature_rate_sum_avg)

        #print("Index: " + str(index) + " | Feature Rating: " + str(feature_rate_sum_avg))

        """
        OTHER AREA
        """
        img = orig_img[930:980, 720:850]
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

        #maxima = average_spec[60:130] # - HERE IS THE ZOOM
        maxima = average_spec[40:90]
        maxima = np.array(maxima)
        spike_index = max(average_spec)

        #maxima = np.array(average_spec)
        #peaks, _ = find_peaks(maxima, height=(max(average_spec)-110))
        peaks, properties = find_peaks(maxima, height=(max(average_spec)-110), prominence=4.5, distance=3.0) #This prominence was hand taylored
        #peaks, properties = find_peaks(maxima, height=(max(average_spec)-110))
        prominences = peak_prominences(maxima, peaks)[0]
        print(peaks)
        #properties["prominences"].max()
        #peaks = [9,13,24]
        #peaks = [8,9,10,12,13,14,23,24,25]

        #print(peaks) #Grabs X values of peaks

        #Now we need to do some manipulation of the peaks and maxima section
        #Peaks -> array of found peaks
        #Maxima -> array of maxima values, maxima[peak] gets peak values

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

        index += 1
        print("Left: " + str(feature_rate_sum_left))
        print("Right: " + str(feature_rate_sum_right))
        feature_rate_sum_avg = (feature_rate_sum_left + feature_rate_sum_right) / 2

        #testing method to take the max of the two instead of the sum
        #feature_rate_sum_avg = max([feature_rate_sum_left, feature_rate_sum_right])

        features_list.append(feature_rate_sum_avg)

        print("Index: " + str(index) + " | Feature Rating: " + str(feature_rate_sum_avg))

    #Doing template output stuff here:
    print("Min Feature Rate: " + str(round(min(features_list),3)))
    print("Max Feature Rate: " + str(round(max(features_list),3)))



    #m,b = np.polyfit(sheldon_scales, features_list,1)
    #m,b = np.polyfit([5,68], [min(features_list), max(features_list)], 1)
    #p0, p1, p2 = np.polyfit(features_list, sheldon_scales, 2)
    #trend = np.polyfit(features_list, sheldon_scales, 3)
    #trend = np.polyfit(sheldon_scales, features_list, 3)

    sheldon_scales = np.array(sheldon_scales)
    features_list = np.array(features_list)
    """
    plt.scatter(sheldon_scales, features_list)
    #plt.plot(sheldon_scales, m*sheldon_scales+b)
    trendpoly = np.polyval(trend, sheldon_scales)
    plt.plot(sheldon_scales, trendpoly)
    plt.title('Harmonic Differences vs. Sheldon Scale w/ Inpainting')
    plt.xlabel("Sheldon Scale")
    plt.ylabel("Harmonic Differences from Min")
    plt.ylim([35, 170])
    plt.show()
    """
    
    print(sheldon_scales)
    plt.scatter(features_list, sheldon_scales)
    #trendpoly = np.poly1d(trend) 
    #trendpoly = np.polyval(trend, features_list)
    #plt.plot(features_list,trendpoly(features_list))
    #plt.plot(features_list, trendpoly)
    x = np.linspace(47, 118, 100) #just for graphing to be easier
    trend = [0.00014162340303767857, -0.05219235501283228, 6.332637857746728, -187.17387181557189]

    plt.plot(x, x*x*x*trend[0] + x*x*trend[1] + x*trend[2] + trend[3])
    #plt.plot(features_list, p0 + ((features_list)**p1) + (features_list*p2))
    plt.title('Harmonic Differences vs. Sheldon Scale w/ Inpainting')
    plt.xlabel("Harmonic Differences from Min")
    plt.ylabel("Sheldon Scales")
    plt.show()
    
    #print("Extracted Slope: " + str(m))
    #print("Extracted Intercept: " + str(b))
    print("Extreacted p0: " + str(trend[0]))
    print("Extreacted p1: " + str(trend[1]))
    print("Extreacted p2: " + str(trend[2]))
    print("Extreacted p3: " + str(trend[3]))

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

    

def Grade_Morgan():

    name = '61.jpg'
    #name = input("Input Image Name: ")#
    
    orig_img = cv2.imread('C:/Users/pm4ti/Desktop/Senior Design/Main Project Code/CoinCherrypicker/Images/MSD_Reverse/' + name,0)
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

    avg_rating = (Bleft_wing_val+BRight_wing_val+Uleft_wing_val+Uright_wing_val)/4
    print("File: " + name + " | Overall Feature Rating: " + str(round(avg_rating, 3)))
    


def overallMSDPlot():
    images = loadImages('gray') #WONT WORK ANYMORE CAUSE OPENER WAS CHANGED
    features_list = []
    #sheldon_scales = [3,4,8,12,15,20,25,30,35,40,45,50,53,55,61,62,63,64,65,67,68,69,70]
    sheldon_scales = [3,4,12,20,25,53,55,62,63,68,69,70]
    index = 0
    name = 'MSD.jpg'
    for orig_img in images:

        #MOVED TO gradeMorganByArea - Inpainting
        #mask1 = cv2.threshold(orig_img, 200, 255, cv2.THRESH_BINARY)[1]
        #orig_img = cv2.inpaint(orig_img, mask1, 0.1, cv2.INPAINT_TELEA)


        Bleft_wing_val = gradeMorganByArea(orig_img, "BL", name)
        #print("File: " + name + " | Feature Rating BL: " + str(Bleft_wing_val))

        BRight_wing_val = gradeMorganByArea(orig_img, "BR", name)
        #print("File: " + name + " | Feature Rating BR: " + str(BRight_wing_val))

        Uleft_wing_val = gradeMorganByArea(orig_img, "UL", name)
        #print("File: " + name + " | Feature Rating UL: " + str(Uleft_wing_val))

        Uright_wing_val = gradeMorganByArea(orig_img, "UR", name)
        #print("File: " + name + " | Feature Rating UR: " + str(Uright_wing_val))

        
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
        #avg_rating = (Bleft_wing_val+BRight_wing_val)/2

        features_list.append(avg_rating)

        index += 1

        print("Index: " + str(index) + " | Overall Feature Rating: " + str(avg_rating))

    #trend = np.polyfit(sheldon_scales, features_list,2)
    trend = np.polyfit(features_list, sheldon_scales,2)

    sheldon_scales = np.array(sheldon_scales)

    #plt.scatter(sheldon_scales, features_list)
    plt.scatter(features_list, sheldon_scales)
    #plt.plot(sheldon_scales, m*sheldon_scales+b)
    plt.title('Harmonic Differences vs. Sheldon Scale')
    plt.ylabel("Sheldon Scale")
    plt.xlabel("Harmonic Differences from Min")
    plt.xlim([0, 125])
    x = np.linspace(26, 88, 100)
    plt.plot(x, (x*x*trend[0] +  x*trend[1] + trend[2]))

    plt.show()
    print(trend)
    print("Max: " + str(round(max(features_list),3)))
    print("Min: " + str(round(min(features_list),3)))
    #print("Extracted Slope: " + str(m))
    #print("Extracted Intercept: " + str(b))

        
    
    

    


if __name__ == '__main__':
    #Transform_VG08_Wheat()
    #Transform_MS67_Wheat()
    #Transform_MS67_Memorial()
    #Numpy_Memorial_Transform()
    #Numpy_VG08_Transform()
    #Numpy_MS67_Transform()
    #Overall_Fourier_to_Sheldon()
    #Template_Overlay()

    #p1 = multiprocessing.Process(target=Numpy_VG08_Transform)
    #p2 = multiprocessing.Process(target=Numpy_MS67_Transform)
    #p3 = multiprocessing.Process(target=Numpy_Memorial_Transform)
    #p4 = multiprocessing.Process(target=Overall_Fourier_to_Sheldon_Wheat)
    #p5 = multiprocessing.Process(target=Overall_Fourier_to_Sheldon_Memorial)
    #p6 = multiprocessing.Process(target=Giga_Opener_Wheat_Overall_Freq_Peak)

    #name = input("Enter name to grade: ") 

    #p7 = multiprocessing.Process(target=Numpy_MS40_Transform)
    #p8 = multiprocessing.Process(target=Given_Image_Name_Grade, args=(name,))
    #p9 = multiprocessing.Process(target=Overall_Derivative_Plot)

    #p10 = multiprocessing.Process(target=Grade_Morgan)
    p11 = multiprocessing.Process(target=overallMSDPlot)

    #p1.start()
    #p2.start()
    #p3.start()
    #p4.start()
    #p5.start()
    #p6.start()
    #p7.start()
    #p8.start()
    #p9.start()

    #p10.start()
    p11.start()
  
