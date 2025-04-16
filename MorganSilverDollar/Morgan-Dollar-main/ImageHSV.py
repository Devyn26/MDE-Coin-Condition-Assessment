'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 3/05/2025
'''
import cv2

import numpy

import ImageOpener
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from buildDatabase import compileHSVData


class KNearestNeighbor(object):
    """
     * The object class KNearestNeighbor is used for ML prediction training
     *
     * So the idea behind the KNearestNeighbor algorithm is to classify an unknown element given provided classification
     * data from elements already analyzed.  In terms of machine learning algorithms it's not too scary, but it's still
     * machine learning.
     *
     * In other words, the algorithm uses feature similarity to predict new data values. It takes the data provided from
     * a training set, analyzes it, and then attempts to determine a value for the new data based on its similarities
     * to those of the training set.
    """

    def __init__(self):
        pass

    # Values are initialized privately to the object
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k):
        # Gets the number of test points found within the test set
        num_test = X.shape[0]
        # This was apparently not used
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        """
         * The first step in KNearestNeighbors is to find the distance/value-range between the test data point and each 
         * training data point.  
         *
         * In this case the distance between each point is being determined using Euclidean distance where X is the 
         * test point and self.X_train contains the testing points.
        """
        # I can't say that I fully understand why this is done but I'm sure there's a reason
        d1 = -2 * np.dot(X, self.X_train.T)
        """
         * The squared value of each element within the testing set is added together with elements of corresponding
         * indices.
         *
         * axis=1 signifies to sum up the columns
         *
         * [[2, 3], [8, 10]] --> [5, 18]
        """
        d2 = np.sum(np.square(X), axis=1, keepdims=True)
        # The same idea as above applies here
        d3 = np.sum(np.square(self.X_train), axis=1)
        # Add and square root to get final distance list
        dist = np.sqrt(d1 + d2 + d3)

        # Initializes an array the same size as the test set for saving results
        y_pred = np.zeros(num_test)
        # Loop for as many tests as there are
        for i in range(num_test):
            """
             * The reason for the k value in the algorithm is to set a limit on how many training points to consider
             * when formulating a conclusion.  Given that the k value is currently set to 5, this means that the 5 
             * closest training points to the testing point are considered for evaluating the result.
            """
            dist_k_min = np.argsort(dist[i])[:k]
            """
             * Based on the closest points within the X_training set, the corresponding colors are identified
            """
            y_kclose = self.y_train[dist_k_min]
            """
             * Based on the k-data points selected for consideration (the nearest neighbors), the most frequently 
             * occurring color is selected as the prediction.
            """
            y_pred[i] = np.argmax(np.bincount(y_kclose))

        # Returns the prediction list
        return y_pred


def Histogram_drawing(HSV_Value):
    plt.subplot(1, 3, 1)
    plt.hist(HSV_Value[0], bins=80, range=(0, 360), histtype='stepfilled', color='r', label='Hue')
    plt.title("Hue")
    plt.xlabel("Degree 0-360")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(HSV_Value[1], bins=40, range=(0, 100), histtype='stepfilled', color='g', label='Saturation')
    plt.title("Saturation")
    plt.xlabel("Value 0-100")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(HSV_Value[2], bins=40, range=(0, 100), histtype='stepfilled', color='b', label='Value')
    plt.title("Value")
    plt.xlabel("value 0-100")
    plt.ylabel("Frequency")

    plt.legend()
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Calculate the Mean and Median HSV of the Image
def mean_median_value(HSV_Value, mode, scale):
    if mode not in ["median", "mean", "both"]:
        print("Invalid mode")
        return None, None
    elif scale > 1 or scale < 0:
        print("Invalid scale")
        return None, None

    # Presumably, all three lists within HSV_Value should be the same length
    HSV_length = math.floor(len(HSV_Value[0]) * scale)

    mean_H, mean_S, mean_V = 0, 0, 0
    median_H, median_S, median_V = 0, 0, 0
    if mode != "median":
        # get the mean value
        mean_H = statistics.mean(HSV_Value[0][:HSV_length])
        mean_S = statistics.mean(HSV_Value[1][:HSV_length])
        mean_V = statistics.mean(HSV_Value[2][:HSV_length])

    if mode != "mean":
        # get the median value
        median_H = statistics.median(HSV_Value[0][:HSV_length])
        median_S = statistics.median(HSV_Value[1][:HSV_length])
        median_V = statistics.median(HSV_Value[2][:HSV_length])

    if mode == "median":
        return [median_H, median_S, median_V], None
    elif mode == "mean":
        return None, [mean_H, mean_S, mean_V]
    else:
        return [median_H, median_S, median_V], [mean_H, mean_S, mean_V]


def Calculate_HSV(image):
    """
     * In the function below RGB is converted to HSV using a set of formulas.  The
     * HSV values for each pixel within the image is calculated starting from (x=0, y=0) to
     * (x=width-1, y=height-1)
     *
     * Param:   Image - resulting RGB matrix obtained from cv2.imread()
     *
     * Return:  HSV_value - A list of 3 lists, the first list contains all hue values, the second contains all
     *                      saturation values, and the third contains all value values.
    """

    # Obtains the width and height of the region to run calculations on
    width = image.shape[0]
    height = image.shape[1]
    # Initialization of lists for hue, saturation, and value values
    lst_hue, lst_sat, lst_val = [], [], []
    # Read RGB for every pixel
    for w in range(width):
        for h in range(height):
            RGB = image[w, h]
            # When read from an image, RGB is read in the order of Blue, Green, Red
            B, G, R = RGB
            B, G, R = float(B), float(G), float(R)
            # This will skip over any black pixels (mask pixels)
            if R == 0 and G == 0 and B == 0:
                continue
            # finds the smallest value between Red, Green, and Blue
            C_low = float(min(RGB))
            # finds the largest value between Red, Green, and Blue
            C_high = float(max(RGB))
            # finds the difference between the max and min
            C_range = float(C_high - C_low)
            """
             * Just as a note: 
             *    - Hue is the angle on the color wheel
             *    - Saturation is the distance from the center of the color wheel (further is greater saturation)
             *    - Value is the depth within the color wheel
            """
            # If there is not max or min, then the calculations are unnecessary
            if C_low == C_high:
                H_hsv_degree = 0
            else:
                H_prime = 0

                # Formula for when Red is the highest
                if C_high == R:
                    H_prime = (G - B) / C_range
                # Formula for when Green is the highest
                elif C_high == G:
                    H_prime = 2.0 + (B - R) / C_range
                # Formula for when Blue is the highest
                elif C_high == B:
                    H_prime = 4.0 + (R - G) / C_range
                """
                 * Multiply result by 60 to scale to 360 degrees.  If the result was negative, add 360 to get the 
                 * equivalent position angle.
                """
                if H_prime < 0:
                    H_hsv = (H_prime * 60) + 360
                else:
                    H_hsv = H_prime * 60

                H_hsv_degree = round(H_hsv, 3)

            # find saturation and Value
            C_high_bin = C_high / 255
            C_range_bin = C_range / 255
            # Find the value, the value is determined by the color with the greatest strength in RGB
            V = round(C_high_bin * 100, 3)
            # Find the saturation by applying a conversion formula, if all RGB values are the same then there is no sat
            if C_low == C_high:
                Saturation = int(0)
            else:
                Saturation = round((C_range_bin / C_high_bin) * 100, 3)
            lst_hue.append(H_hsv_degree)
            lst_sat.append(Saturation)
            lst_val.append(V)
    # Creates a new list composed of all HSV values
    HSV_val = [lst_hue, lst_sat, lst_val]
    return HSV_val


def color_classifications(average_HSV):
    """
     * This thing is scary and it should be put into an excel sheet.  I believe this is a list of predetermined HSV
     * mean values from other coins analyzed.
    """

    Coin_135 = [[34.9, 37.2, 28.3], [27.5, 48.1, 28.5], [41.2, 45.8, 26.4], [36.9, 60.8, 30.6], [45.1, 45.5, 32.3],
                [43.9, 52.5, 29.3], [36.9, 52.7, 31.6], [43.1, 49.6, 30.0], [36.9, 61.5, 30.5], [45.1, 47.4, 26.7],
                [42.7, 60.7, 32.4], [40.4, 57.7, 30.9], [49.4, 48.9, 29.1], [42.0, 49.0, 31.1], [33.7, 43.7, 25.5],
                [36.5, 42.1, 29.3], [42.0, 47.5, 30.4], [36.1, 39.8, 26.8], [35.3, 53.0, 28.5], [41.2, 46.2, 30.5],
                [38.4, 38.5, 29.1], [28.2, 64.9, 30.0], [42.0, 51.5, 30.0], [31.4, 60.8, 28.2], [35.3, 37.4, 22.7],
                [27.5, 46.8, 30.0], [40.0, 55.3, 30.6], [38.8, 43.1, 31.1], [45.9, 45.9, 27.1], [26.3, 25.0, 24.0],
                [45.9, 45.9, 27.1], [35.7, 46.0, 29.0], [51.8, 31.9, 31.3], [42.4, 42.0, 29.4], [38.4, 39.5, 35.2],
                [49.4, 41.5, 23.5], [62.7, 57.8, 22.3], [69.4, 71.4, 27.8], [45.1, 46.7, 27.7], [48.2, 73.5, 28.0],
                [53.3, 54.5, 27.1], [48.2, 52.0, 19.1], [53.3, 57.7, 23.4], [53.3, 56.2, 25.3], [45.5, 64.6, 26.4],
                [39.2, 74.4, 23.4], [45.5, 64.6, 26.4], [60.4, 46.8, 26.3], [51.8, 57.9, 25.6], [42.7, 67.6, 24.7],
                [36.1, 43.7, 35.4], [42.7, 59.8, 29.1], [45.1, 71.6, 27.0], [45.1, 62.0, 29.6], [29.0, 41.7, 21.3],
                [63.1, 52.5, 28.2], [58.8, 42.5, 20.0], [55.3, 45.9, 21.7], [76.9, 46.7, 33.6], [70.2, 61.1, 27.0],
                [58.4, 48.8, 27.1], [61.6, 56.8, 29.6], [52.5, 64.9, 29.0], [76.9, 80.1, 25.5], [78.4, 49.3, 30.0],
                [60.8, 65.1, 23.4], [43.5, 58.6, 29.3], [64.3, 69.0, 25.9], [77.3, 68.4, 30.0], [43.5, 63.1, 27.5],
                [44.7, 52.5, 28.0], [54.9, 57.1, 28.2], [87.8, 66.0, 25.8], [41.2, 78.7, 22.4], [62.7, 55.4, 27.3],
                [42.7, 46.5, 27.7], [73.3, 58.8, 29.2], [73.3, 63.6, 29.1], [66.3, 76.7, 26.5], [74.9, 54.7, 26.5],
                [84.7, 58.7, 30.3], [55.3, 64.6, 28.2], [51.0, 69.3, 23.8], [55.7, 77.7, 27.8], [60.4, 61.6, 28.0],
                [56.1, 74.9, 26.5], [85.5, 65.7, 26.2], [72.9, 62.3, 27.2], [82.4, 69.2, 26.5], [58.0, 66.0, 26.2],
                [81.2, 62.7, 29.8], [82.0, 80.1, 29.2], [70.6, 62.6, 25.6], [60.0, 65.6, 22.4], [62.4, 67.7, 26.5],
                [58.8, 72.5, 26.5], [71.0, 71.6, 26.7], [68.2, 65.5, 27.9], [79.6, 60.4, 31.6], [78.8, 74.8, 29.3],
                [58.8, 61.6, 25.1], [78.4, 78.7, 25.3], [83.5, 71.1, 28.5], [81.6, 65.5, 26.1], [60.8, 66.7, 24.7],
                [60.8, 51.2, 22.2], [75.3, 67.3, 28.0], [60.0, 57.0, 23.9], [80.4, 79.1, 23.3], [48.6, 73.8, 22.4],
                [67.8, 88.9, 24.6], [72.2, 60.6, 27.2], [67.8, 60.5, 24.9], [57.6, 67.0, 22.6], [55.7, 74.0, 26.5],
                [72.5, 75.9, 24.9], [87.8, 58.1, 32.2], [71.0, 57.3, 29.1], [63.9, 62.7, 23.8], [51.0, 72.1, 28.8],
                [83.1, 80.4, 20.4], [52.5, 60.5, 24.8], [60.0, 61.3, 23.9], [69.0, 60.2, 26.5], [71.4, 69.2, 24.2],
                [71.8, 84.8, 23.1], [87.8, 81.3, 26.9], [72.2, 71.4, 23.1], [80.4, 79.1, 23.3], [55.7, 73.7, 23.5],
                [67.8, 88.9, 24.6], [38.4, 45.2, 31.2], [42.7, 75.2, 36.3], [72.2, 70.6, 30.3], [56.1, 88.9, 24.3],
                average_HSV]

    """
     * I'm not sure where this comes from, and it seems a bit redundant given the loop following the initialization
     * of this array.  I don't see why this couldn't be an array of strings instead, but maybe I'm missing something.  
     * Either way something like this should be moved to an excel sheet.
    """
    colors_Label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                    1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                    1, 2, 2, 0]  # 66#tail65        #tail60

    # Turns out this is actually unused
    color_list = []
    for file in colors_Label:
        if file == 2:
            color_list.append("red")
        elif file == 1:
            color_list.append("orange")
        elif file == 0:
            color_list.append("brown")
    """
     * So this section of the code is actually a training set for a machine learning nearest-neighbor algorithm.  
     * Thanks to these descriptive names and lengthy comments provided I don't really know what the hell is going on 
     * but I'll try my best.
    """
    # First, the giant list of data is converted to a numpy array type (in this case it is more like a matrix)
    X_A = np.array(Coin_135)

    """
     * X_A.T is the transposed matrix of X_A, numpy takes this matrix and translates it into a covariance matrix
     *
     * Ok, so a covariance matrix holds the covariance values of select pairs of elements within the matrix.
     *
     * Covariance is basically a measurement of how much two (typically random) variables vary together.  
     *
     * An easier way to think about covariance is to think of the average slope found within a scatter-plot, if the 
     * data points form to what looks to be a positive slop, there is a positive covariance, if there is a negative 
     * slope then its negative covariance, and if a direction can't be determined the covariance is close to zero.
     *
     * Anyways, the reason this is important for this application is to get the eigenvalues and eigenvectors
     * of the distribution.
    """

    np.cov(X_A.T)
    # Similarly to the Coin_135 matrix, color labels is turned into an numpy array type
    # Not sure why this is here

    X_B = X_A

    y = np.array(colors_Label)
    X_train, X_test, y_train, y_test = train_test_split(X_A, y, test_size=(1/136), random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(X_train)
    print(X_test)

    kneigh = KNeighborsClassifier(n_neighbors=5)
    kneigh.fit(X_train, y_train)
    y_test = kneigh.predict(X_test)
    print(y_test)

    return
    """
     * So given that we all had to take linear algebra I'm assuming everyone has at least heard of eigenvalues and
     * eigenvectors.
     *
     * Just as a reminder in case needed, eigenvectors point in the direction in which a matrix is altered by a 
     * transformation, in a sense it is "stretched" with the transformation, the eigenvalue is the scale at which it
     * is stretched.
     * 
     * In this case, the eigenvectors of the covariance matrix point in the general direction of the data distribution
     * and the eigenvalues are determined by the overall length between the distributions points.
    """
   # eigvalue, eigvector = np.linalg.eig(np.cov(X_A.T))
    """
     * 
     * All numpy hstack does is basically rearrange arrays to be sorted column-wise
     * 
     * In this case, it's used to essential remove one of the eigenvectors from the set obtained above.  To be honest
     * I don't exactly know what the purpose of this is aside from being able to obtain a dot product that only has two 
     * columns.  This is likely part of some algorithm/formula they implemented.
     * 
    """
    #a = np.hstack((eigvector[:, 0].reshape(3, -1), eigvector[:, 1].reshape(3, -1)))
    """
     * Once again, I'm not sure why this is being done, but all this is is every element of the Coin_135 matrix 
     * subtracted by the average value over the entire matrix
    """
    #X_A = X_A - X_A.mean(axis=0)
    """
     * Here the dot product between the eigenvectors (eigenmatrix?) and the newly modified Coin_135 matrix is taken to 
     * produce a 2 column matrix which will be used as a model for nearest-neighbor prediction training.
    """
    #X_new1 = X_A.dot(a)

    # Creates a 136x2 matrix containing only values of 200
    positive = np.ones((136, 2))
    positive = positive * 200

    # I don't know why this is done, but I'm sure there was a reason, perhaps it is to guarantee no negative values
    X = X_new1 + positive
    y = Color_Label

    """
     * The training set for the algorithm is composed of all of the data originally initialized in Coin_135 matrix 
     * and the color labels array.
    """
    X_train = X[:135, :]
    y_train = y[:]
    """
     * The test set for the prediction, this is the last array of the matrix, a.k.a the average HSV values calculated 
     * for the image that was provided as a parameter for the function.
    """
    X_test = X[135:136, :]
    # A new object of KNearestNeighbor type is initialized as KNN
    KNN = KNearestNeighbor()
    # The training sets are added as values to the algorithm
    KNN.train(X_train, y_train)
    # The KNN algorithm is ran, an explanation is given in the object class defining the algorithm
    y_pred = KNN.predict(X_test, k=5)

    # The numerical prediction values are given meaning by the conditional statement
    results = " "
    if y_pred == 0:
        results = "Dark Gray"
    elif y_pred == 1:
        results = "Gray"
    else:
        results = "Silver"
    # ----------Outliers-------------START
    outliers = 0

    """
     * I believe this conditional chain is essentially searching for training data points that are an exceptional
     * distance from the testing data point, hence why they are referred to as outliers.
     * 
     * This will probably need to be redone as it seems to be hardcoded to specific training data points within the set
    """
    X_test = X[135:136, :]
    for i in range(10):
        X1 = X[134 - i:135 - i, :]
        # This is just standard Euclidean distance
        dist = np.sqrt(np.sum(np.square(X_test - X1)))
        if dist < 1:
            if i == 0 or i == 5:
                outliers = 1  # "Red Maybe"
            elif i == 2:
                outliers = 1  # "Colorful"
            elif i == 3:
                outliers = 1  # "Cleaned"
            elif i == 4:
                outliers = 1  # "Pretty toning"
            elif i == 6:
                outliers = 1  # "Great Luster"
            elif i == 8:
                outliers = 1  # "MS67"
            elif i == 9:
                outliers = 1  # "Questionable"
            elif i == 7:
                outliers = 1  # "Cheery red"
            elif i == 1:
                outliers = 1  # "golden"
            break
        else:
            outliers = 0
    if outliers:
        results2 = "; Possible Outlier"
    else:
        results2 = " "

    # -------------Outliers-------------END
    Final_Output = [results, results2]
    return Final_Output


def mask_and_in_paint_image(image):
    # Gets the width & height of the image
    h, w = image.shape[:2]
    # Radius includes the entire coin except for the edge ridges
    radius = 460
    # Initialize a new mask as an empty image
    maskCircle = np.zeros_like(image)
    # Gets half width and height of the image
    h_floor, w_floor = h // 2, w // 2
    # Create a circular mask to apply to each image
    maskCircle = cv2.circle(maskCircle, (h_floor, w_floor), radius, (255, 255, 255), -1)
    # Applies the circular mask to the image
    maskedImage = cv2.bitwise_and(image, maskCircle)
    """
     * Image In-painting
    """
    # Converts image to grayscale
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Applies a binary threshold to the image on the range
    thresholdMask = cv2.threshold(grayImg, 200, 240, cv2.THRESH_BINARY)[1]
    # thresholdMask = cv2.adaptiveThreshold(grayImg, 240, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    """
     * Just as a general description, in-painting is usually used to
     * remove scratches/noise from images by applying a mask over the
     * areas of interest, in this case I'm not exactly sure why this was here
     * and not exactly sure what it does.
     *
     * Also, this takes some time between 30 - 45 seconds to complete so if the code
     * freezes here for a bit that's why.  In-paint is not designed for this kind of
     * implementation, so there should be a better way to accomplish this.
     *
     * 3rd parameter is the in-paint radius
    """
    in_paintedImage = cv2.inpaint(maskedImage, thresholdMask, .1, cv2.INPAINT_TELEA)

    return maskedImage


def get_color_percent(HSV_Value):
    """
     * Used to find the specified percentage of color found within the coin
     *
     * Param:   HSV_Value - A list of 3 lists, the first list contains all hue values, the second contains all saturation
     *                      values, and the third contains all value values.
     * Return:  List - A list of size 2 containing the old_percentage_red value and the new_percentage_red value
    """

    # HMerge = HSV_Value_r1[0] + HSV_Value_r2[0] + HSV_Value_r3[0] + HSV_Value_r4[0] + HSV_Value_r5[0] + HSV_Value_r6[0]
    # SMerge = HSV_Value_r1[1] + HSV_Value_r2[1] + HSV_Value_r3[1] + HSV_Value_r4[1] + HSV_Value_r5[1] + HSV_Value_r6[1]
    # VMerge = HSV_Value_r1[2] + HSV_Value_r2[2] + HSV_Value_r3[2] + HSV_Value_r4[2] + HSV_Value_r5[2] + HSV_Value_r6[2]

    hsv_medians, _ = mean_median_value(HSV_Value, "median", 1)

    """
     * This is where they calculated the % red of the coin.  Obviously this is completely unusable for us but it's
     * a good reference for if/when we implement a %silver measurement.
    """

    # Old Percent Red Equation Calculations
    # Old_Percentage_Red = (((Median_S+Median_V)-(Min_S+Min_V))/((Max_S+Max_V)-(Min_S+Min_V)))*100
    # Old_Percentage_Red =round(Old_Percentage_Red,1)

    # New Percent Red Equation Calculations
    New_Percentage_Red = ((2 * hsv_medians[1] * hsv_medians[2]) / (hsv_medians[1] + hsv_medians[2])) * \
                         math.exp(-((hsv_medians[0] - 26) / 8) ** 2)
    New_Percentage_Red = round(New_Percentage_Red, 1)

    # print('Possible Red Percentage: {0:.1f}'.format(Old_Percentage_Red))
    # print('Possible New Red Percentage: {0:.1f}'.format(New_Percentage_Red))

    return [None, New_Percentage_Red]


def Image_HSV(numImages, classify):
    # Initializes a list of 3 lists to contain the median values for each element of hsv
    hsv_medians = []
    # Initializes a list of 2 lists, the first containing a the old percent red, the second containing the new
    percent_values = []
    # A list to hold the elements of the coin description
    """
     * Year, Grader, Grade Type, WebGrade, Inventory #, Tail Feather #, Mint Location
    """
    descriptionList = numpy.empty((7, 0)).tolist()
    # Gets the image matrices from ImageOpener and the file names associated
    imgList, imgNames = ImageOpener.loadImages('color')

    """
     * This loop goes through all of the image matrices obtained from cv2.read in ImageOpener.  Each image is first 
     * masked to show only the important parts
    """
    if numImages == 'All':
        numImages = len(imgList)
    for i in range(numImages):
        # This is to skip over the reverse, consider to remove later on
        modified_image = mask_and_in_paint_image(imgList[i])


        # return
        """
         * This is used for initializing specific regions of the coin to be separately analyzed for their HSV values,
         * in our case we are not doing this to start but may later on.  Regions in the image are indexed 
         * by [y1:y2, x1:x2]
        """
        # regions_list = [modified_image[450:600, 700:850], modified_image[400:500, 110:260],
        #                 modified_image[620:720, 130:240], modified_image[310:380, 120:280],
        #                 modified_image[310:380, 700:880], modified_image[220:600, 670:730]]

        # get HSV value
        t1 = time.perf_counter()
        HSV_Value = Calculate_HSV(modified_image)
        t2 = time.perf_counter()

        # This is where the histogram could be developed
        Histogram_drawing(HSV_Value)

        # Calculate percentage of color and medians
        percents = get_color_percent(HSV_Value)
        percent_values.append(percents)

        # Color classification (PCA -> KNearestNeighbors)
        _, mean_values = mean_median_value(HSV_Value, "mean", 1)
        average_HSV = [round(mean_values[2], 1), round(mean_values[1], 1), round(mean_values[0], 1)]
        if classify:
            color_classifications(average_HSV)

        compileHSVData(average_HSV, percents, None, imgNames[i])
        # Write image information to database

        # END OF LOOP


# Get the average HSV of a single coin
def getAvgHSV(image):
    # coinImg.load('C:/Users/lizzi/Documents/fall2022/srdes/morganImages/Morgan 2021-CC MS70 obverse.jpg')

    # Masks image
    # Gets the width & height of the image
    h, w = image.shape[:2]
    # Radius includes the entire coin except for the edge ridges
    radius = 460
    # Initialize a new mask as an empty image
    maskCircle = np.zeros_like(image)
    # Gets half width and height of the image
    h_floor, w_floor = h // 2, w // 2
    # Create a circular mask to apply to each image
    maskCircle = cv2.circle(maskCircle, (h_floor, w_floor), radius, (255, 255, 255), -1)
    # Applies the circular mask to the image
    maskedImage = cv2.bitwise_and(image, maskCircle)

    # Initializes a list of 3 lists to contain the median values for each element of hsv
    hsv_medians = []
    # Initializes a list of 2 lists, the first containing a the old percent red, the second containing the new
    percent_values = []
    # A list to hold the elements of the coin description
    """
     * Year, Grader, Grade Type, WebGrade, Inventory #, Tail Feather #, Mint Location
    """
    descriptionList = numpy.empty((7, 0)).tolist()

    # get HSV value
    t1 = time.perf_counter()
    HSV_Value = Calculate_HSV(maskedImage)
    t2 = time.perf_counter()

    # Calculate percentage of color and medians
    percents = get_color_percent(HSV_Value)
    percent_values.append(percents)

    # Color classification (PCA -> KNearestNeighbors)
    _, mean_values = mean_median_value(HSV_Value, "mean", 1)
    median_values, _ = mean_median_value(HSV_Value, "median", 1)
    average_HSV = [round(mean_values[2], 1), round(mean_values[1], 1), round(mean_values[0], 1)]
    median_HSV = [median_values[2], median_values[1], median_values[0]]
    print(average_HSV)
    return mean_values

# Calculate the Mean and Median HSV of the Image at different percentage
def mean_median_value2(im_hsv):
    # get the mean value
    hue_mean = statistics.mean(im_hsv[0])
    saturation_mean = statistics.mean(im_hsv[1])
    value_mean = statistics.mean(im_hsv[2])
    # get the median value
    h,s,l=[],[],[]
    h = im_hsv[0]
    s = im_hsv[1]
    v = im_hsv[2]
    h.sort()
    s.sort()
    v.sort()
    length=int(len(h)/2)
    hue_median = h[length]
    saturation_median = s[length]
    value_median = v[length]
    # print('MEAN        hue: {0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_mean, saturation_mean, value_mean))
    # print('MEDIUM      hue: {0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_median, saturation_median,value_median))

    # 25% median value
    new_data_hue = h[:length]
    new_data_sat = s[:length]
    new_data_val = v[:length]
    length_25=int(length/2)

    hue_25median = new_data_hue[length_25]
    sat_25median = new_data_sat[length_25]
    val_25median = new_data_val[length_25]
    # print('25% MEDIUM  hue: {0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_25median, sat_25median,val_25median))
    # 25% mean value
    hue_25mean = statistics.mean(new_data_hue)
    sat_25mean = statistics.mean(new_data_sat)
    val_25mean = statistics.mean(new_data_val)
    # print('25% MEAN    hue :{0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_25mean, sat_25mean, val_25mean))

    # 75% median value
    new_data_hue = h[length:]
    new_data_sat = s[length:]
    new_data_val = v[length:]

    hue_75median = new_data_hue[length_25]
    sat_75median = new_data_sat[length_25]
    val_75median = new_data_val[length_25]
    # print('75% MEDIUM  hue: {0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_75median, sat_75median,val_75median))
    # 75% mean value
    hue_75mean = statistics.mean(new_data_hue)
    sat_75mean = statistics.mean(new_data_sat)
    val_75mean = statistics.mean(new_data_val)
    # print('75% MEAN    hue :{0:.1f}, saturation: {1:.1f}, Value: {2:.1f}'.format(hue_75mean, sat_75mean, val_75mean))
    # print('')
    # print('')


if __name__ == '__main__':
    # Put 'All' for first parameter to read all images
    getAvgHSV()
    # Image_HSV(numImages='All', classify=False)
