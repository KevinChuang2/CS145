# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import random as rd
import math


# insert an all-one column as the first column
def addAllOneColumn(matrix):
    n = matrix.shape[0]  # total of data points
    p = matrix.shape[1]  # total number of attributes

    newMatrix = np.zeros((n, p + 1))
    newMatrix[:, 0] = np.ones(n)
    newMatrix[:, 1:] = matrix

    return newMatrix


# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    y = dataframe['y']
    x = dataframe.drop('y', axis=1)
    return x, y


# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# compute average logL
def compute_avglogL(x, y, beta):
    eps = 1e-50
    n = y.shape[0]
    logL = 0.0
    for i in range(n): 
        logL += (np.dot(y[i] , sigmoid(np.dot(x[i],(beta)))) - np.log(1 + eps + sigmoid(np.dot(x[i],(beta)))))
    return logL/n


# predicted_y and y are the predicted and actual y values respectively as numpy arrays
# function prints the accuracy
def compute_accuracy(predicted_y, y):
    acc = 100.0
    acc = np.sum(predicted_y == y) / predicted_y.shape[0]
    return acc


# train_x and train_y are numpy arrays
# lr (learning rate) is a scalar
# function returns value of beta calculated using (0) batch gradient descent
def getBeta_BatchGradient(train_x, train_y, lr, num_iter, verbose):
    beta = np.random.rand(train_x.shape[1])

    n = train_x.shape[0]  # total of data points
    p = train_x.shape[1]  # total number of attributes

    beta = np.random.rand(p)
    derivLogL = np.zeros(p)
    for iter in range (0, num_iter):
        pib = sigmoid(np.dot(train_x, beta))
        derivLogL = np.dot(np.transpose(train_x), pib-train_y)
        beta -= lr * derivLogL

    return beta


# train_x and train_y are numpy arrays
# function returns value of beta calculated using (1) Newton-Raphson method
def getBeta_Newton(train_x, train_y, num_iter, verbose):
    n = train_x.shape[0]  # total of data points
    p = train_x.shape[1]  # total number of attributes

    beta = np.random.rand(p)

    for iter in range(0, num_iter):
        pib = sigmoid(np.dot(train_x, beta))
        derivLogL = np.dot(np.transpose(train_x), train_y - pib)
        secDerivLogL = np.identity(p)
        for i in range(0,p):
            for j in range(0,p):
                secDerivLogL[i][j] -= np.dot(np.dot(np.transpose(train_x[:,i]), np.transpose(train_x[:,j])), np.dot(pib, 1-pib))
        beta -= np.dot(np.linalg.inv(secDerivLogL), derivLogL)
        if (verbose == True and iter % 1000 == 0):
            avgLogL = compute_avglogL(train_x, train_y, beta)
            print(f'average logL for iteration {iter}: {avgLogL} \t')
    return beta

def getBeta_RegularizedBatchGradient(train_x, train_y, lr, num_iter, lambdaVar , verbose):
    beta = np.random.rand(train_x.shape[1])

    n = train_x.shape[0]  # total of data points
    p = train_x.shape[1]  # total number of attributes

    beta = np.random.rand(p)
    derivLogL = np.zeros(p)
    for iter in range (0, num_iter):
        pib = sigmoid(np.dot(train_x, beta))
        derivLogL = np.dot(np.transpose(train_x), pib-train_y) - 2*lambdaVar*beta
        beta -= lr * derivLogL

    return beta

# Linear Regression implementation
class LogisticRegression(object):
    # Initializes by reading data, setting hyper-parameters, and forming linear model
    # Forms a linear model (learns the parameter) according to type of beta (0 -  batch gradient, 1 - Newton-Raphson)
    # Performs z-score normalization if isNormalized is 1
    # Print intermidate training loss if verbose = True
    def __init__(self, lr=0.003, num_iter=10000, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.verbose = verbose
        self.train_x = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.test_x = pd.DataFrame()
        self.test_y = pd.DataFrame()
        self.algType = 0
        self.isNormalized = 0

    def load_data(self, train_file, test_file):
        self.train_x, self.train_y = getDataframe(train_file)
        self.test_x, self.test_y = getDataframe(test_file)

    def normalize(self):
        # Applies z-score normalization to the dataframe and returns a normalized dataframe
        self.isNormalized = 1
        data = np.append(self.train_x, self.test_x, axis=0)
        means = data.mean(0)
        std = data.std(0)
        self.train_x = (self.train_x - means).div(std)
        self.test_x = (self.test_x - means).div(std)

    # Gets the beta according to input
    def train(self, algType):
        self.algType = algType
        newTrain_x = addAllOneColumn(self.train_x.values)  # insert an all-one column as the first column
        if (algType == '0'):
            beta = getBeta_BatchGradient(newTrain_x, self.train_y.values, self.lr, self.num_iter, self.verbose)
            print('Beta: ', beta)

        elif (algType == '1'):
            beta = getBeta_Newton(newTrain_x, self.train_y.values, self.num_iter, self.verbose)
            print('Beta: ', beta)
        elif (algType == '2'):
            beta = getBeta_RegularizedBatchGradient(newTrain_x, self.train_y.values, self.lr, self.num_iter, 9.2, self.verbose)
            print('Beta: ', beta)
        else:
            print('Incorrect beta_type! Usage: 0 - batch gradient descent, 1 - Newton-Raphson method, 2 - Regularized Batch Gradient')

        predicted_y = newTrain_x.dot(beta)
        train_avglogL = compute_avglogL(newTrain_x, self.train_y.values, beta)
        print('Training avgLogL: ', train_avglogL)

        return beta

    # Predicts the y values of all test points
    # Outputs the predicted y values to the text file named "logistic-regression-output_algType_isNormalized" inside "output" folder
    # Computes accuracy
    def predict(self, beta):
        newTest_x = addAllOneColumn(self.test_x.values)
        self.predicted_y = (sigmoid(newTest_x.dot(beta)) >= 0.5)
        n = newTest_x.shape[0]
        output = np.zeros((n, 2))
        output[:, 0] = self.test_y
        output[:, 1] = self.predicted_y
        np.savetxt(
            'output/logistic-regression-output' + '_' + str(self.algType) + '_' + str(self.isNormalized) + '.txt',
            output, delimiter='\t', newline='\n')
        accuracy = compute_accuracy(self.predicted_y, self.test_y.values)
        return accuracy


if __name__ == '__main__':
    # Change 1st paramter to 0 for batch gradient, 1 for newton's method
    # Add a second paramter with value 1 for z score normalization
    algType = sys.argv[1]
    isNormalized = sys.argv[2]
    print('Learning Algorithm Type: ', algType)
    print('Is normalization used: ', isNormalized)
    
    logisticM = LogisticRegression()
    logisticM.load_data('logistic_regression_train.csv', 'logistic_regression_test.csv')
    # do we need normalization?
    if (isNormalized == '1'):
        logisticM.normalize()

    # training
    beta = logisticM.train(algType)

    # testing
    test_accuracy = logisticM.predict(beta)
    print('Test accuracy: ', test_accuracy)
