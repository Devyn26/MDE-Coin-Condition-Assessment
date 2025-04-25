


#Temp file in order to plot the points of the HSV coins
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D

def makeHuePlot():

    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]
    BN_Hue_Mean = [29.6, 23.8, 28.9]
    RB_Hue_Mean = [27.3, 27.1, 21.0]
    RD_Hue_Mean = [25.0, 29.8, 24.7, 29.0, 26.4, 30.1]
    QC_Hue_Mean = [21.0]

    overallHue = BN_Hue_Mean + RB_Hue_Mean + RD_Hue_Mean + QC_Hue_Mean
    overallX = BNX + RBX + RDX + QCX
    x = np.array(overallX)
    m, b = np.polyfit(overallX, overallHue, 1)

    plt.title('Hue Averages')
    plt.xlabel('Coin Numbers')
    plt.ylabel('Average Hue')
    plt.scatter(BNX,BN_Hue_Mean,  c ='brown')        # plot x and y using default line style and color
    plt.scatter(RBX, RB_Hue_Mean,  c ='orange')
    plt.scatter(RDX, RD_Hue_Mean,  c ='red')
    plt.scatter(QCX, QC_Hue_Mean,  c ='purple')
    #plt.plot(x, m*x + b)
    plt.show()


  
def makeSatPlot():

    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]

    BN_Sat_Mean = [32.5, 29.0, 24.4]
    RB_Sat_Mean = [59.3, 38.9, 38.7]
    RD_Sat_Mean = [76.8, 52.3, 50.4, 51.5, 59.3, 59.3]
    QC_Sat_Mean = [78.7]

    overallSat = BN_Sat_Mean + RB_Sat_Mean + RD_Sat_Mean + QC_Sat_Mean
    overallX = BNX + RBX + RDX + QCX

    x = np.array(overallX)
    m, b = np.polyfit(overallX, overallSat, 1)

    plt.title('Saturation Averages')
    plt.xlabel('Coin Numbers')
    plt.ylabel('Average Sat')
    plt.scatter(BNX,BN_Sat_Mean,  c ='brown')        # plot x and y using default line style and color
    plt.scatter(RBX, RB_Sat_Mean,  c ='orange')
    plt.scatter(RDX, RD_Sat_Mean,  c ='red')
    plt.scatter(QCX, QC_Sat_Mean,  c ='purple')
    plt.plot(x, m*x + b)
    plt.show()


def makeValPlot():

    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]

    BN_Val_Mean = [50.3, 35.6, 38.3]
    RB_Val_Mean = [47.0, 54.4, 49.8]
    RD_Val_Mean = [77.9, 78.4, 62.7, 45.0, 67.6, 75.4]
    QC_Val_Mean = [64.7]

    overallVal = BN_Val_Mean + RB_Val_Mean + RD_Val_Mean + QC_Val_Mean
    overallX = BNX + RBX + RDX + QCX

    x = np.array(overallX)
    m, b = np.polyfit(overallX, overallVal, 1)


    plt.title('Value Averages')
    plt.xlabel('Coin Numbers')
    plt.ylabel('Average Val')
    plt.scatter(BNX,BN_Val_Mean,  c ='brown')        # plot x and y using default line style and color
    plt.scatter(RBX, RB_Val_Mean,  c ='orange')
    plt.scatter(RDX, RD_Val_Mean,  c ='red')
    plt.scatter(QCX, QC_Val_Mean,  c ='purple')
    plt.plot(x, m*x + b)
    plt.show()


def SVPlot():
    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]
   

    BN_Hue_Mean = [29.6, 23.8, 28.9]
    RB_Hue_Mean = [27.3, 27.1, 21.0]
    RD_Hue_Mean = [25.0, 29.8, 24.7, 29.0, 26.4, 30.1]
    QC_Hue_Mean = [21.0]
    BN_Sat_Mean = [32.5, 29.0, 24.4]
    RB_Sat_Mean = [59.3, 38.9, 38.7]
    RD_Sat_Mean = [76.8, 52.3, 50.4, 51.5, 59.3, 59.3]
    QC_Sat_Mean = [78.7]
    BN_Val_Mean = [50.3, 35.6, 38.3]
    RB_Val_Mean = [47.0, 54.4, 49.8]
    RD_Val_Mean = [77.9, 78.4, 62.7, 45.0, 67.6, 75.4]
    QC_Val_Mean = [64.7]


    overallHue = BN_Hue_Mean + RB_Hue_Mean + RD_Hue_Mean + QC_Hue_Mean
    overallVal = BN_Val_Mean + RB_Val_Mean + RD_Val_Mean + QC_Val_Mean
    overallSat = BN_Sat_Mean + RB_Sat_Mean + RD_Sat_Mean + QC_Sat_Mean

    averageHue = sum(overallHue)/len(overallHue)
    minHue = min(overallHue)
    maxHue = max(overallHue)

    print("//--------GLARE NOT REMOVED--------//")
    print("Average Hue: ", averageHue)
    print("Hue Range: ", minHue, " - ", maxHue)


    x = np.array(overallSat)
    m, b = np.polyfit(overallSat, overallVal, 1)

    plt.xlim([0, 100])
    plt.ylim([0, 100])
    #plt.legend([averageHue, minHue, maxHue], ['Average Hue', 'Min Hue', 'Max Hue'])
    #plt.text("Hue Range: ", minHue,"-",maxHue)
    plt.title('S vs V Plot')
    plt.xlabel('Average Sat')
    plt.ylabel('Average Val')
    plt.scatter(BN_Sat_Mean,BN_Val_Mean, c ='brown')        # plot x and y using default line style and color
    plt.scatter(RB_Sat_Mean, RB_Val_Mean, c='orange')
    plt.scatter(RD_Sat_Mean, RD_Val_Mean, c='red')
    plt.scatter(QC_Sat_Mean, QC_Val_Mean,c="purple")
    plt.plot(x, m*x + b)
    plt.show()














def makeHuePlotGlare():

    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]
    BN_Hue_Mean = [29.6, 23.8, 29.0]
    RB_Hue_Mean = [27.4, 26, 21.9]
    RD_Hue_Mean = [24.7, 29.8, 24.3, 29.4, 26.4, 30.8]
    QC_Hue_Mean = [19.4]

    overallHue = BN_Hue_Mean + RB_Hue_Mean + RD_Hue_Mean + QC_Hue_Mean
    overallX = BNX + RBX + RDX + QCX
    x = np.array(overallX)
    m, b = np.polyfit(overallX, overallHue, 1)

    plt.title('Hue Averages without Glare')
    plt.xlabel('Coin Numbers')
    plt.ylabel('Average Hue')
    plt.scatter(BNX,BN_Hue_Mean,  c ='brown')        # plot x and y using default line style and color
    plt.scatter(RBX, RB_Hue_Mean,  c ='orange')
    plt.scatter(RDX, RD_Hue_Mean,  c ='red')
    plt.scatter(QCX, QC_Hue_Mean,  c ='purple')
    #plt.plot(x, m*x + b)
    plt.show()


  
def makeSatPlotGlare():

    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]

    BN_Sat_Mean = [48.1, 47.0, 39.3]
    RB_Sat_Mean = [73.7, 56.1, 54.9]
    RD_Sat_Mean = [83.5, 47.2, 64.2, 64.4, 72.2, 64.2]
    QC_Sat_Mean = [87.4]

    overallSat = BN_Sat_Mean + RB_Sat_Mean + RD_Sat_Mean + QC_Sat_Mean
    overallX = BNX + RBX + RDX + QCX

    x = np.array(overallX)
    m, b = np.polyfit(overallX, overallSat, 1)

    plt.title('Saturation Averages without Glare')
    plt.xlabel('Coin Numbers')
    plt.ylabel('Average Sat')
    plt.scatter(BNX,BN_Sat_Mean,  c ='brown')        # plot x and y using default line style and color
    plt.scatter(RBX, RB_Sat_Mean,  c ='orange')
    plt.scatter(RDX, RD_Sat_Mean,  c ='red')
    plt.scatter(QCX, QC_Sat_Mean,  c ='purple')
    plt.plot(x, m*x + b)
    plt.show()


def makeValPlotGlare():

    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]


    BN_Val_Mean = [49.9, 35.0, 37.5]
    RB_Val_Mean = [46.1, 55.0, 53.1]
    RD_Val_Mean = [72.9, 77.9, 61.8, 45.0, 69.9, 80.6]
    QC_Val_Mean = [64.3]

    overallVal = BN_Val_Mean + RB_Val_Mean + RD_Val_Mean + QC_Val_Mean
    overallX = BNX + RBX + RDX + QCX

    x = np.array(overallX)
    m, b = np.polyfit(overallX, overallVal, 1)


    plt.title('Value Averages without Glare')
    plt.xlabel('Coin Numbers')
    plt.ylabel('Average Val')
    plt.scatter(BNX,BN_Val_Mean,  c ='brown')        # plot x and y using default line style and color
    plt.scatter(RBX, RB_Val_Mean,  c ='orange')
    plt.scatter(RDX, RD_Val_Mean,  c ='red')
    plt.scatter(QCX, QC_Val_Mean,  c ='purple')
    plt.plot(x, m*x + b)
    plt.show()


def SVPlotGlare():
    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]
   

    BN_Hue_Mean = [29.6, 23.8, 29.0]
    RB_Hue_Mean = [27.4, 26, 21.9]
    RD_Hue_Mean = [24.7, 29.8, 24.3, 29.4, 26.4, 30.8]
    QC_Hue_Mean = [19.4]




    BN_Sat_Mean = [32.5, 29.0, 24.4]
    RB_Sat_Mean = [59.3, 38.9, 38.7]
    RD_Sat_Mean = [76.8, 52.3, 50.4, 51.5, 59.3, 59.3]
    QC_Sat_Mean = [78.7]
    BN_Val_Mean = [50.3, 35.6, 38.3]
    RB_Val_Mean = [47.0, 54.4, 49.8]
    RD_Val_Mean = [77.9, 78.4, 62.7, 45.0, 67.6, 75.4]
    QC_Val_Mean = [64.7]


    BN_Sat_Mean = [48.1, 47.0, 39.3]
    RB_Sat_Mean = [73.7, 56.1, 54.9]
    RD_Sat_Mean = [83.5, 47.2, 64.2, 64.4, 72.2, 64.2]
    QC_Sat_Mean = [87.4]
    BN_Val_Mean = [49.9, 35.0, 37.5]
    RB_Val_Mean = [46.1, 55.0, 53.1]
    RD_Val_Mean = [72.9, 77.9, 61.8, 45.0, 69.9, 80.6]
    QC_Val_Mean = [64.3]


    BN_Hue_Mean = [31.4,29.6,23.8,29.0,27.2,30.4]
    RB_Hue_Mean = [27.4,26.0,19.5,21.9,24.0,25.9]
    BN_Sat_Mean = [61.2,48.1,47.0,39.3,47.3,36.3]
    RB_Sat_Mean = [73.7,56.5,47.8,54.9,57.3,65.6]
    BN_Val_Mean = [43.2,49.9,35.0,37.5,44.3,55.0]
    RB_Val_Mean = [46.1,55.0,43.6,53.1,56.9,41.8]



    overallHue = BN_Hue_Mean + RB_Hue_Mean + RD_Hue_Mean + QC_Hue_Mean
    overallVal = BN_Val_Mean + RB_Val_Mean + RD_Val_Mean + QC_Val_Mean
    overallSat = BN_Sat_Mean + RB_Sat_Mean + RD_Sat_Mean + QC_Sat_Mean


    averageHue = sum(overallHue)/len(overallHue)
    minHue = min(overallHue)
    maxHue = max(overallHue)
    print("//--------GLARE REMOVED--------//")
    print("Average Hue: ", averageHue)
    print("Hue Range: ", minHue, " - ", maxHue)


    x = np.array(overallSat)
    m, b = np.polyfit(overallSat, overallVal, 1)

    plt.xlim([0, 100])
    plt.ylim([0, 100])

    plt.title('S vs V Plot without Glare')
    plt.xlabel('Average Sat')
    plt.ylabel('Average Val')
    plt.scatter(BN_Sat_Mean,BN_Val_Mean, c ='brown')        # plot x and y using default line style and color
    plt.scatter(RB_Sat_Mean, RB_Val_Mean, c='orange')
    plt.scatter(RD_Sat_Mean, RD_Val_Mean, c='red')
    plt.scatter(QC_Sat_Mean, QC_Val_Mean,c="purple")
    plt.plot(x, m*x + b)
    plt.show()







def HSV3D():
    BNX = [0,1,2]
    RBX = [3,4,5]
    RDX = [6,7,8,9,10,11]
    QCX = [12]
   

    BN_Hue_Mean = [29.6, 23.8, 29.0]
    RB_Hue_Mean = [27.4, 26, 21.9]
    RD_Hue_Mean = [24.7, 29.8, 24.3, 29.4, 26.4, 30.8]
    QC_Hue_Mean = [19.4]


    BN_Sat_Mean = [32.5, 29.0, 24.4]
    RB_Sat_Mean = [59.3, 38.9, 38.7]
    RD_Sat_Mean = [76.8, 52.3, 50.4, 51.5, 59.3, 59.3]
    QC_Sat_Mean = [78.7]
    BN_Val_Mean = [50.3, 35.6, 38.3]
    RB_Val_Mean = [47.0, 54.4, 49.8]
    RD_Val_Mean = [77.9, 78.4, 62.7, 45.0, 67.6, 75.4]
    QC_Val_Mean = [64.7]


    BN_Sat_Mean = [48.1, 47.0, 39.3]
    RB_Sat_Mean = [73.7, 56.1, 54.9]
    RD_Sat_Mean = [83.5, 47.2, 64.2, 64.4, 72.2, 64.2]
    QC_Sat_Mean = [87.4]
    BN_Val_Mean = [49.9, 35.0, 37.5]
    RB_Val_Mean = [46.1, 55.0, 53.1]
    RD_Val_Mean = [72.9, 77.9, 61.8, 45.0, 69.9, 80.6]
    QC_Val_Mean = [64.3]

    BN_Hue_Mean = [31.4,29.6,23.8,29.0,27.2,30.4]
    RB_Hue_Mean = [27.4,26.0,19.5,21.9,24.0,25.9]
    BN_Sat_Mean = [61.2,48.1,47.0,39.3,47.3,36.3]
    RB_Sat_Mean = [73.7,56.5,47.8,54.9,57.3,65.6]
    BN_Val_Mean = [43.2,49.9,35.0,37.5,44.3,55.0]
    RB_Val_Mean = [46.1,55.0,43.6,53.1,56.9,41.8]

    overallHue = BN_Hue_Mean + RB_Hue_Mean + RD_Hue_Mean + QC_Hue_Mean
    overallVal = BN_Val_Mean + RB_Val_Mean + RD_Val_Mean + QC_Val_Mean
    overallSat = BN_Sat_Mean + RB_Sat_Mean + RD_Sat_Mean + QC_Sat_Mean


    averageHue = sum(overallHue)/len(overallHue)
    minHue = min(overallHue)
    maxHue = max(overallHue)
    print("//--------GLARE REMOVED--------//")
    print("Average Hue: ", averageHue)
    print("Hue Range: ", minHue, " - ", maxHue)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.array(overallSat)
    m, b = np.polyfit(overallSat, overallVal, 1)

    plt.xlim([0, 100])
    plt.ylim([0, 100])

    plt.title('H vs S vs V Plot without Glare')
    plt.xlabel('Average Sat')
    plt.ylabel('Average Val')
    ax.set_zlabel('Average Hue')

    ax.scatter(BN_Sat_Mean,BN_Val_Mean,BN_Hue_Mean, c ='brown')        # plot x and y using default line style and color
    ax.scatter(RB_Sat_Mean, RB_Val_Mean, RB_Hue_Mean, c='orange')
    ax.scatter(RD_Sat_Mean, RD_Val_Mean, RD_Hue_Mean, c='red')
    ax.scatter(QC_Sat_Mean, QC_Val_Mean,QC_Hue_Mean,c="purple")
    #ax.plot_surface(BN_Sat_Mean,BN_Val_Mean,BN_Hue_Mean, cmap ='viridis', edgecolor ='green')
    #plt.plot(x, m*x + b)
    plt.show()






def SVPlotGlare_NEWREGIONS():
    BNX = [0,1,2,3,4,5]
    RBX = [6,7,8,9,10,11]
    RDX = [12,13,14,15,16,17]
    QCX = [18]
   

    BN_Hue_Mean = [29.6, 23.8, 29.0]
    RB_Hue_Mean = [27.4, 26, 21.9]
    RD_Hue_Mean = [24.7, 29.8, 24.3, 29.4, 26.4, 30.8]
    QC_Hue_Mean = [19.4]




    BN_Sat_Mean = [32.5, 29.0, 24.4]
    RB_Sat_Mean = [59.3, 38.9, 38.7]
    RD_Sat_Mean = [76.8, 52.3, 50.4, 51.5, 59.3, 59.3]
    QC_Sat_Mean = [78.7]
    BN_Val_Mean = [50.3, 35.6, 38.3]
    RB_Val_Mean = [47.0, 54.4, 49.8]
    RD_Val_Mean = [77.9, 78.4, 62.7, 45.0, 67.6, 75.4]
    QC_Val_Mean = [64.7]


    BN_Sat_Mean = [48.1, 47.0, 39.3]
    RB_Sat_Mean = [73.7, 56.1, 54.9]
    RD_Sat_Mean = [83.5, 47.2, 64.2, 64.4, 72.2, 64.2]
    QC_Sat_Mean = [87.4]
    BN_Val_Mean = [49.9, 35.0, 37.5]
    RB_Val_Mean = [46.1, 55.0, 53.1]
    RD_Val_Mean = [72.9, 77.9, 61.8, 45.0, 69.9, 80.6]
    QC_Val_Mean = [64.3]


    BN_Hue_Mean = [31.4,29.6,23.8,29.0,27.2,30.4]
    RB_Hue_Mean = [27.4,26.0,19.5,21.9,24.0,25.9]
    BN_Sat_Mean = [61.2,48.1,47.0,39.3,47.3,36.3]
    RB_Sat_Mean = [73.7,56.5,47.8,54.9,57.3,65.6]
    BN_Val_Mean = [43.2,49.9,35.0,37.5,44.3,55.0]
    RB_Val_Mean = [46.1,55.0,43.6,53.1,56.9,41.8]



    overallHue = BN_Hue_Mean + RB_Hue_Mean + RD_Hue_Mean + QC_Hue_Mean
    overallVal = BN_Val_Mean + RB_Val_Mean + RD_Val_Mean + QC_Val_Mean
    overallSat = BN_Sat_Mean + RB_Sat_Mean + RD_Sat_Mean + QC_Sat_Mean


    averageHue = sum(overallHue)/len(overallHue)
    minHue = min(overallHue)
    maxHue = max(overallHue)
    print("//--------GLARE REMOVED--------//")
    print("Average Hue: ", averageHue)
    print("Hue Range: ", minHue, " - ", maxHue)


    x = np.array(overallSat)
    m, b = np.polyfit(overallSat, overallVal, 1)

    plt.xlim([0, 100])
    plt.ylim([0, 100])

    plt.title('S vs V Plot without Glare')
    plt.xlabel('Average Sat')
    plt.ylabel('Average Val')
    plt.scatter(BN_Sat_Mean,BN_Val_Mean, c ='brown')        # plot x and y using default line style and color
    plt.scatter(RB_Sat_Mean, RB_Val_Mean, c='orange')
    plt.scatter(RD_Sat_Mean, RD_Val_Mean, c='red')
    plt.scatter(QC_Sat_Mean, QC_Val_Mean,c="purple")
    plt.plot(x, m*x + b)
    plt.show()


if __name__ == '__main__':

    h = multiprocessing.Process(target=makeHuePlot, args=())
    s = multiprocessing.Process(target=makeSatPlot, args=())
    v = multiprocessing.Process(target=makeValPlot, args=())
    makeSVPlot = multiprocessing.Process(target=SVPlot, args=())

    #h.start()
    #s.start()
    #v.start()
    #makeSVPlot.start()

    h = multiprocessing.Process(target=makeHuePlotGlare, args=())
    s = multiprocessing.Process(target=makeSatPlotGlare, args=())
    v = multiprocessing.Process(target=makeValPlotGlare, args=())
    makeSVPlot = multiprocessing.Process(target=SVPlotGlare, args=())

    #h.start()
    #s.start()
    #v.start()
    makeSVPlot.start()

    hsv3DPlot = multiprocessing.Process(target=HSV3D, args=())
    hsv3DPlot.start()


