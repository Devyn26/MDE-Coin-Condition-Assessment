"""
MorganGrader.py

Processes all features from a provided training dataset (csv); predicts a grade using a Multilayer Perceptron
Classifier or Regressor

Author: Jasper Emick, Lizzie LaVallee
Date: 10 Mar 2023
"""
import numpy as np
# import tensorflow as tf
import pandas as pd
#from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError

#from plotConditionData import scatterPlot
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
# from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from scipy.stats import linregress
import matplotlib.pyplot as plt

# Good seeds: 10, 30, 40, 200, 370, 620
SEED_REG = 30
SEED_SPLIT = 200
# What

class Grader:
    def __init__(self):
        self.database = "MorganSilverDollar/Morgan_Dollar_main/image_database.csv"
        self.model = MLPRegressor(solver='adam',
                                  random_state=SEED_REG,
                                  hidden_layer_sizes=(8, 6),
                                  max_iter=1000,
                                  activation='relu')
        self.processedData = {}
        self.scaler = StandardScaler()
        self.confidence = MLPRegressor(solver='adam',
                                  random_state=SEED_REG,
                                  hidden_layer_sizes=(8, 6),
                                  max_iter=1000,
                                  activation='relu')

    def PreProcessing(self, features,
                      label='Grade', test_size=0.05):
        """ Prepares data for the grading model to train and predict """

        df = pd.read_csv(self.database)
        X = df[features].values
        y = df[label].values
        inventory = df[['Inventory #']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED_SPLIT)

        # Standard scale for optimization and to protect against
        sc = self.scaler
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        self.processedData = {'TrainingData': X_train, 'TestingData': X_test, 'TrainingLabels': y_train,
                              'TestingLabels': y_test, 'Inventory': inventory}

    def TrainModel(self):
        self.model.fit(self.processedData['TrainingData'], self.processedData['TrainingLabels'])

    def SaveModel(self, filename='MorganSilverDollar/Morgan_Dollar_main/model.sav'):
        pickle.dump(self.model, open(filename, 'wb'))

    def LoadModel(self, filename='MorganSilverDollar/Morgan_Dollar_main/model.sav'):
        self.model = pickle.load(open(filename, 'rb'))

    def TrainConfidenceModel(self):
        self.confidence.fit(self.processedData['TrainingData'], self.processedData['TrainingLabels'])

    def SaveConfidenceModel(self, filename='MorganSilverDollar/Morgan_Dollar_main/confidenceModel.sav'):
        pickle.dump(self.confidence, open(filename, 'wb'))

    def LoadConfidenceModel(self, filename='MorganSilverDollar/Morgan_Dollar_main/confidenceModel.sav'):
        self.confidence = pickle.load(open(filename, 'rb'))
        
    def PredictGrade(self, inputCoin):
        X_test = self.scaler.transform(inputCoin)
        try:
            prediction = self.model.predict(X_test)
        except NotFittedError:
            self.TrainModel()
            prediction = self.model.predict(X_test)
        if prediction[0] > 70:
            return 70.0
        return round(prediction[0], 1)
    
    def CalculateConfidence(self, grade):
        X_train, X_test, y_train, y_test, inventory = PreProcessing_Testing()

        predictions = np.zeros([len(X_test)])

        clf = MLPRegressor(solver='adam',
                       random_state=SEED_REG,
                       hidden_layer_sizes=(8, 6),
                       max_iter=1000,
                       activation='relu')

        # ANN.fit(tf.expl_train, y_train, epochs=10)

        # Weight coins to attempt to make a uniform distribution in the database to prevent skewing

        clf.fit(X_train, y_train)

        numWithin5 = 0
        numPerfect = 0
        for x in range(len(X_test)):
            # man = ANN.predict([X_test[x]])
            man = clf.predict([X_test[x]])
            predictions[x] = man[0]
            if y_test[x] - 5 <= man[0] <= y_test[x] + 5:
                numWithin5 += 1
            if y_test[x] == round(man[0]):
                numPerfect += 1

        # predictions[predictions < 55] = 55.0
        predictions[predictions > 70] = 70.0
        differences = np.subtract(predictions, y_test)
        differences[differences > 10] = 10
        differences[differences < -10] = -10

        adjusted_predictions = np.zeros([len(X_test)])
        for x in range(len(X_test)):
            if (differences[x] > 0):
                adjusted_predictions[x] = predictions[x] - (abs(differences[x]) * 2)
            else:
                adjusted_predictions[x] = predictions[x]

        testPercents = np.divide(adjusted_predictions[adjusted_predictions > 0], y_test[y_test > 0])

        #if np.sum(abs(differences) <= 1) / len(X_test) > 0.55:
        coefficients = np.polyfit(y_test, testPercents, 2)
        quadratic_model = np.poly1d(coefficients)
        return round((coefficients[0] * (grade * grade)) + (coefficients[1] * grade) + coefficients[2], 2) * 100


def PreProcessing_Testing():
    df = pd.read_csv("MorganSilverDollar/Morgan_Dollar_main/image_database.csv")

    # Standard Database
    X = df[[
        'EdgeFreq Flat Obverse',
        'EdgeFreq Flat Reverse',
        'EdgeFreq RedOrange Obverse',
        'EdgeFreq RedOrange Reverse',
        'EdgeFreq Yellow Obverse',
        'EdgeFreq Yellow Reverse',
        'EdgeFreq Green Obverse',
        'EdgeFreq Green Reverse',
        'Brilliance Obverse',
        'Brilliance Reverse',
        'Toning Obverse',
        'Toning Reverse'
    ]].values
    inventory = df[['Inventory #']]
    y = df.iloc[:, 3].values

    # Splits data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=SEED_SPLIT)

    # Standard scale for optimization and to protect against
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, inventory


def TheMachineIsLearning():

    X_train, X_test, y_train, y_test, inventory = PreProcessing_Testing()

    predictions = np.zeros([len(X_test)])

    clf = MLPRegressor(solver='adam',
                       random_state=SEED_REG,
                       hidden_layer_sizes=(8, 6),
                       max_iter=1000,
                       activation='relu')

    # ANN.fit(tf.expl_train, y_train, epochs=10)

    # Weight coins to attempt to make a uniform distribution in the database to prevent skewing

    clf.fit(X_train, y_train)
    # clf.fit(X_train, y_train)

    numWithin5 = 0
    numPerfect = 0
    for x in range(len(X_test)):
        # man = ANN.predict([X_test[x]])
        man = clf.predict([X_test[x]])
        predictions[x] = man[0]
        if y_test[x] - 5 <= man[0] <= y_test[x] + 5:
            numWithin5 += 1
        if y_test[x] == round(man[0]):
            numPerfect += 1

    percents = numWithin5 / len(X_test)
    perfect = numPerfect / len(X_test)

    # predictions[predictions < 55] = 55.0
    predictions[predictions > 70] = 70.0
    print(predictions)
    differences = np.subtract(predictions, y_test)
    differences[differences > 10] = 10
    differences[differences < -10] = -10
    print(differences)
    print("+/-7:", np.sum(abs(differences) <= 7) / len(X_test))
    print("+/-5:", np.sum(abs(differences) <= 5) / len(X_test))
    print("+/-3:", np.sum(abs(differences) <= 3) / len(X_test))
    print("+/-1:", np.sum(abs(differences) <= 1) / len(X_test))

    adjusted_predictions = np.zeros([len(X_test)])
    for x in range(len(X_test)):
        if (differences[x] > 0):
            adjusted_predictions[x] = predictions[x] - (abs(differences[x]) * 2)
        else:
            adjusted_predictions[x] = predictions[x]

    testPercents = np.divide(adjusted_predictions[adjusted_predictions > 0], y_test[y_test > 0])
    slope, intercept, r_value, p_value, std_err = linregress(y_test, testPercents)

    #if np.sum(abs(differences) <= 1) / len(X_test) > 0.55:
    this, that = train_test_split(inventory, test_size=0.05, random_state=SEED_SPLIT)

    ran = np.arange(len(y_test), 0, -1)
    # scatterPlot(differences, ran, y_test, "Error Range for Test Coins",
    #              "Margin of Error (Predicted Grade - Original Grade)", np.transpose(np.array(that))[0])
    
    # scatterPlot(y_test, testPercents, y_test, "Percent Accuracy of Coins",
    #              "Actual Coin Grade", np.transpose(np.array(that))[0])
    # print("Slope: ", slope)
    # print("Intercept:", intercept)
    # print("Standard Error: ", std_err)
    # print("R^2 Value:", r_value * r_value)
    # print("P Value:", p_value)

    # plt.scatter(y_test, testPercents, color='blue', label='Data points')
    # plt.plot(y_test, [slope * xi + intercept for xi in y_test], color='red', label=f'Best fit line: y = {slope:.2f}x + {intercept:.2f}')
    # plt.legend()
    # plt.show()

    coefficients = np.polyfit(y_test, testPercents, 2)
    quadratic_model = np.poly1d(coefficients)

    # x_pred = np.linspace(min(y_test), max(y_test), 100)
    # y_pred = quadratic_model(x_pred)
    # print("Quadratic Equation: y = {:.2f}x^2 + {:.2f}x + {:.2f}".format(*coefficients))
    # plt.scatter(y_test, testPercents, color='blue', label='Data Points')
    # plt.plot(x_pred, y_pred, color='red', label='Quadratic Fit')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Quadratic Regression')
    # plt.legend()
    # plt.show()
    # ANN.predict



if __name__ == '__main__':
    #TheMachineIsLearning()
    g = Grader()
    g.PreProcessing(features=['EdgeFreq Flat Obverse',
                              'EdgeFreq Flat Reverse',
                              'EdgeFreq RedOrange Obverse',
                              'EdgeFreq RedOrange Reverse',
                              'EdgeFreq Yellow Obverse',
                              'EdgeFreq Yellow Reverse',
                              'EdgeFreq Green Obverse',
                              'EdgeFreq Green Reverse',
                              'Brilliance Obverse',
                              'Brilliance Reverse',
                              'Toning Obverse',
                              'Toning Reverse'
                              ])
    g.TrainModel()
    g.SaveModel()
    g.TrainConfidenceModel()
    g.SaveConfidenceModel()
    # g.LoadModel()
    # # Get the prediction of the first coin in TestingData, rounded to 1 decimal:
    # print(g.PredictGrade(g.processedData['TestingData'][0].reshape(1, -1)))

    # Get the +/-5 accuracy of the testing data:
    # predictions = []
    # for coin in g.processedData['TestingData']:
    #     # print(round(g.PredictGrade(coin.reshape(1, -1)), 1))
    #     predictions.append(round(g.PredictGrade(coin.reshape(1, -1)), 1))
    # differences = g.processedData['TestingLabels'] - predictions
    # print(np.sum(abs(differences) <= 5) / len(g.processedData['TestingLabels']))
