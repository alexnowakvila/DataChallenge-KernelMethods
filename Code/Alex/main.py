import os
import numpy as np
from utils import *
from svm import *
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split

np.random.seed(2)  # set random seed
dataset = 0
# kernel = 'given_features'
kernel = 'linear'
# kernel = 'spectrum'
solver = 'cvxopt'
# solver = 'mine'
path_data = ("/home/alexnowak/DataChallenge-KernelMethods/Data/")
Dataset = read_data(path_data, dataset=dataset)
# barrier method parameters
t0 = 1.
mu = 3.
tol = 1e-1
LS= True
tau = 1e-6
cut = 1700
test_size = 0.25

if kernel == 'given_features':
  X = Dataset["Xtr_mat50"]
  Y = Dataset["Ytr"]
  X_train, X_test, Y_train, Y_test = (train_test_split(X, Y, 
                                      test_size=test_size))
  X_train, X_test = X_train.T, X_test.T
  # if data not centered add extra dimension
  X_train = np.concatenate((X_train, np.ones((1, X_train.shape[1]))), axis=0)
  X_test = np.concatenate((X_test, np.ones((1, X_test.shape[1]))), axis=0)

  svm_primal = Euclidean_SVM(tau, X_train, Y_train, dual=False)
  svm_dual = Euclidean_SVM(tau, X_train, Y_train, dual=True)

elif kernel == 'linear':
  X = Dataset["Xtr_mat50"]
  Y = Dataset["Ytr"]
  X_train, X_test, Y_train, Y_test = (train_test_split(X, Y, 
                                      test_size=test_size))
  X_train, X_test = X_train.T, X_test.T
  # if data not centered add extra dimension
  X_train = np.concatenate((X_train, np.ones((1, X_train.shape[1]))), axis=0)
  X_test = np.concatenate((X_test, np.ones((1, X_test.shape[1]))), axis=0)

  Kernel = KernelLinear(dim=X_train.shape[0])
  # svm_primal = kernel_SVM(tau, X_train, Y_train, Kernel, dual=False)
  svm_dual = kernel_SVM(tau, X_train, Y_train, Kernel, dual=True)

elif kernel == 'spectrum':
  X = Dataset["Xtr"]
  Y = Dataset["Ytr"]
  X_train, X_test, Y_train, Y_test = (train_test_split(X, Y, 
                                      test_size=test_size))
  Kernel = KernelSpectrum(k=4)
  svm_dual = kernel_SVM(tau, X_train, Y_train, Kernel, dual=True)

else:
  raise ValueError("Kernel {} not implemented".format(kernel))

#############################################################################
#  dual with kernel
#############################################################################

x_sol, alpha, acc_train = svm_dual.svm_solver(solver=solver)
acc_test = svm_dual.compute_accuracy(X_test, Y_test, alpha)
print('acc_train', acc_train)
print('acc_test', acc_test)
print('DUAL OPTIMIZATION FINISHED')

#############################################################################
#  primal with kernel
#############################################################################

# x_sol, alpha, acc_train = svm_primal.svm_solver(solver=solver)
# acc_test = svm_primal.compute_accuracy(X_test, Y_test, alpha)
# print('acc_train', acc_train)
# print('acc_test', acc_test)
# print('PRIMAL OPTIMIZATION FINISHED')


