# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import random as rd

#insert an all-one column as the first column
def addAllOneColumn(matrix):
    n = matrix.shape[0] #total of data points
    p = matrix.shape[1] #total number of attributes
    
    newMatrix = np.zeros((n,p+1))
    newMatrix[:,1:] = matrix
    newMatrix[:,0] = np.ones(n)
    
    return newMatrix
    
# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    y = dataframe['y']
    x = dataframe.drop('y', axis=1)
    return x, y

# train_x and train_y are numpy arrays
# function returns value of beta calculated using (0) the formula beta = (X^T*X)^ -1)*(X^T*Y)
def getBeta(train_x, train_y):   
    n = train_x.shape[0] #total of data points
    p = train_x.shape[1] #total number of attributes    
    
    x_transpose = train_x.transpose()
    xTxInverse = np.linalg.inv(np.dot(x_transpose, train_x))
    beta = np.dot(xTxInverse, x_transpose.dot(train_y))

    return beta
    
# train_x and train_y are numpy arrays
# lr (learning rate) is a scalar
# function returns value of beta calculated using (1) batch gradient descent
def getBetaBatchGradient(train_x, train_y, lr, num_iter):
    beta = np.random.rand(train_x.shape[1])

    n = train_x.shape[0] #total of data points
    p = train_x.shape[1] #total number of attributes

    
    beta = np.random.rand(p)
    #update beta interatively
    for iter in range(0, num_iter):
       deriv = np.zeros(p)
       for i in range(n):
           deriv = deriv + train_x[i] * (np.dot(train_x[i].transpose(), beta) - train_y[i])
       deriv = deriv / n
       beta = beta - lr*deriv
    return beta

# train_x and train_y are numpy arrays
# lr (learning rate) is a scalar
# function returns value of beta calculated using (2) stochastic gradient descent
def getBetaStochasticGradient(train_x, train_y, lr):
    n = train_x.shape[0] #total of data points
    p = train_x.shape[1] #total number of attributes

    beta = np.random.rand(p)

    epoch = 100;
    for iter in range(epoch):
        indices = list(range(n))
        rd.shuffle(indices)
        for i in range(n):
            idx = indices[i]
            term = train_y[idx] - np.dot(np.transpose(train_x[idx]),beta)
            beta = beta + lr * np.dot(term, train_x[idx])
    return beta
    
# predicted_y and test_y are the predicted and actual y values respectively as numpy arrays
# function prints the mean squared error value for the test dataset
def compute_mse(predicted_y, y):
    mse = 100.0
    mse = np.sum((predicted_y - y)**2)/predicted_y.shape[0]
    return mse
    
# Linear Regression implementation
class LinearRegression(object):
    # Initializes by reading data, setting hyper-parameters, and forming linear model
    # Forms a linear model (learns the parameter) according to type of beta (0 - closed form, 1 - batch gradient, 2 - stochastic gradient)
    # Performs z-score normalization if z_score is 1
    def __init__(self,lr=0.001, num_iter=5000):
        self.lr = lr
        self.num_iter = num_iter
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
        means = self.train_x.mean(0)
        std = self.train_x.std(0)
        self.train_x = (self.train_x - means).div(std)
        self.test_x = (self.test_x - means).div(std)
    
    # Gets the beta according to input
    def train(self, algType):
        self.algType = algType
        newTrain_x = addAllOneColumn(self.train_x.values) #insert an all-one column as the first column
        if(algType == '0'):
            beta = getBeta(newTrain_x, self.train_y.values)
            print('Beta: ', beta)
            
        elif(algType == '1'):
            beta = getBetaBatchGradient(newTrain_x, self.train_y.values, self.lr, self.num_iter)
            print('Beta: ', beta)
        elif(algType == '2'):
            beta = getBetaStochasticGradient(newTrain_x, self.train_y.values, self.lr)
            print('Beta: ', beta)
        else:
            print('Incorrect beta_type! Usage: 0 - closed form solution, 1 - batch gradient descent, 2 - stochastic gradient descent')
            
        predicted_y = newTrain_x.dot(beta)
        train_mse = compute_mse(predicted_y, self.train_y.values)
        print('Training MSE: ', train_mse)
        
        return beta
            
    # Predicts the y values of all test points
    # Outputs the predicted y values to the text file named "logistic-regression-output_algType_isNormalized" inside "output" folder
    # Computes MSE
    def predict(self, beta):
        newTest_x = addAllOneColumn(self.test_x.values)
        self.predicted_y = newTest_x.dot(beta)
        n = newTest_x.shape[0]
        output = np.zeros((n,2))
        output[:,0] = self.test_y
        output[:,1] = self.predicted_y
        np.savetxt('output/linear-regression-output' + '_' + str(self.algType) + '_' + str(self.isNormalized) + '.txt', output, delimiter = '\t', newline = '\n')
        mse = compute_mse(self.predicted_y, self.test_y.values)
        return mse
    
    
if __name__ == '__main__':
    # Change 1st paramter to 0 for closed form, 1 for batch gradient, 2 for stochastic gradient
    # Add a second paramter with value 1 for z score normalization
    algType = sys.argv[1]
    isNormalized = sys.argv[2]
    print('Learning Algorithm Type: ', algType)
    print('Is normalization used: ', isNormalized)
    
    lm = LinearRegression()
    lm.load_data('linear-regression-train.csv','linear-regression-test.csv')
    #do we need normalization?    
    if(isNormalized == '1'):
        lm.normalize()
    
    #training
    beta = lm.train(algType)
    
    
    #testing
    test_mse = lm.predict(beta)
    print('Test MSE: ', test_mse)
    