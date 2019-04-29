import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import sys
import pandas as pd

# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    y = dataframe['y']
    x = dataframe.drop('y', axis=1)
    y = y*2 -1.0
    
    return x, y

def compute_accuracy(predicted_y, y):
    acc = 100.0
    acc = np.sum(predicted_y == y)/predicted_y.shape[0]
    return acc

def linear_kernel_point(x1, x2):
    return np.dot(x1, x2)
    
def polynomial_kernel_point(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel_point(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
             
def linear_kernel(X):
    return np.dot(X,X.T)

def polynomial_kernel(X, p=3):
    return (1 + np.dot(X, X.T)) ** p

#
def gaussian_kernel(X, sigma=5.0):
    n = X.shape[0]
    K = np.zeros((n,n))
    X = X.values
    print('Gaussian Kernel computing: I am not efficiently implemented. Please consider smarter implementation')
    for i in range(n):
            for j in range(n):
                K[i,j] = gaussian_kernel_point(X[i], X[j])
    return K

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        if(kernel == linear_kernel):
            self.kernel_point = linear_kernel_point
        elif(kernel == polynomial_kernel):
            self.kernel_point = polynomial_kernel_point
        else:
            self.kernel_point = gaussian_kernel_point
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Kernel matrix
        K = self.kernel(X)
        
        # dealing with dual form quadratic optimization
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples),'d')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv].values
        self.sv_y = y[sv].values

        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept via average calculating b over support vectors
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    # predict labels for test dataset
    def project(self, X):
        if self.w is not None: ## linear case
            predict_y = np.dot(X, np.transpose(self.w) )
            return predict_y + self.b
        else: ## non-linear case
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                temp = 0
                for j in range(len(self.sv)):
                    temp += self.a[j] * self.sv_y[j] * self.kernel_point( self.sv[j], X.iloc[i])
                y_predict[i] = temp
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    # Change 1st parameter to 0 for hard margin, 1 for soft margin
    marginType = sys.argv[1]
    # Change 2nd parameter to 0 for linear_kernel, 1 for polynomial_kernel, and 2 for gaussian_kernel  
    kernelType = sys.argv[2]
    print('Margin: ', marginType)
    print('kernel Type: ', kernelType)
    
    #load data
    train_x, train_y = getDataframe('Data\\train.csv')
    test_x, test_y = getDataframe('Data\\test.csv')
    
    #training    
    C = 500
    if marginType == '0' and kernelType == '0':
        mysvm = SVM()
    elif marginType == '0' and kernelType == '1':
        mysvm = SVM(polynomial_kernel)
    elif marginType == '0' and kernelType == '2':
        mysvm = SVM(gaussian_kernel)
    elif marginType == '1' and kernelType == '0':
        mysvm = SVM(linear_kernel, C)
    elif marginType == '1' and kernelType == '1':
        mysvm = SVM(polynomial_kernel, C)
    elif marginType == '1' and kernelType == '2':
        mysvm = SVM(gaussian_kernel, C)
    else:
        mysvm = SVM()
        print('error, use default svm')
        
    mysvm.fit(train_x, train_y)

    #testing
    predict_y = mysvm.predict(test_x)
    n = test_x.shape[0]
    output = np.zeros((n,2))
    output[:,0] = test_y
    output[:,1] = predict_y
    np.savetxt('output/test' + '_' + str(marginType) + '_' + str(kernelType) + '.txt', output, delimiter = '\t', newline = '\n')
    test_accuracy = compute_accuracy(predict_y, test_y)
    
    print('Test accuracy: ', test_accuracy)
