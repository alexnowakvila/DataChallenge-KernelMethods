import os
import numpy as np
from utils import *
from svm import *
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
from tabulate import tabulate


#############################################################################
#  cross validation
#############################################################################

def cross_validate(K, Y, cut, tau, n_partitions=4):
  # permute
  n = K.shape[0]
  # permute data
  perm = np.random.permutation(n)
  K = K[perm]
  K = K[:, perm]
  Y = Y[perm]
  assert n % n_partitions == 0
  cut = int(n - n/n_partitions)
  perm = np.arange(0, n)
  Acc_train, Acc_test = [], []
  for i in range(n_partitions):
    print("\n ------------- PARTITION NUMBER {} -------------- \n".format(i+1))
    acc_train, acc_test = run(K, Y, cut, perm, tau)
    Acc_train.append(acc_train)
    Acc_test.append(acc_test)
    # change partition
    aux = perm[cut:]
    perm = np.concatenate((aux, perm[:cut]), 0)
  info = [[i, Acc_train[i], Acc_test[i]] for i in range(n_partitions)]
  header = ["Partition", "Acc. Train", "Acc. Test"]
  print("\n", tabulate(info, header))
  mean_train, mean_test = np.mean(Acc_train), np.mean(Acc_test)
  print("\n Mean Train: {}, Mean Test: {} \n"
        .format(mean_train, mean_test))
  return mean_train, mean_test

def evaluate_taus(K, Y, cut, taus, n_partitions=4, kernel=None):
  info, header = [], []
  for tau in taus:
    print("\n EVALUATING TAU = {}\n".format(tau))
    acc_tr, acc_te = cross_validate(K, Y, cut, tau, n_partitions=n_partitions)
    info.append([tau, acc_tr, acc_te])
  header = ["Tau", "Acc. Train", "Acc. Test"]
  print("\n Kernel: {} \n".format(kernel))
  print("\n", tabulate(info, header))

#############################################################################
#  train and test
#############################################################################


def run(K, Y, cut, perm, tau):
  # train test split
  K_train, K_test, Y_train, Y_test = kernel_train_test_split(K, Y, cut, perm=perm)
  # solve
  svm_dual = kernel_SVM(tau, K_train, Y_train, dual=True)
  x_sol, alpha, acc_train = svm_dual.svm_solver(solver=solver)
  # test
  acc_test = svm_dual.compute_accuracy(K_test, Y_test, alpha)
  return acc_train, acc_test


if __name__ == "__main__":
  np.random.seed(1)  # set random seed
  dataset = 0
  # kernel = 'given_features'
  # kernel = 'linear'
  kernel = 'spectrum'
  solver = 'cvxopt'
  # solver = 'mine'
  path_data = ("/home/alexnowak/DataChallenge-KernelMethods/Data/")
  Dataset = read_data(path_data, dataset=dataset)
  # regularization parameter
  tau = 1e-7
  cut = 1500
  test_size = 0.33
  # choose kernel
  if kernel == 'linear':
    X = Dataset["Xtr_mat50"].T
    Y = Dataset["Ytr"]
    Kernel = KernelLinear(dim=X.shape[0])
    K = Kernel.kernel_matrix(X)
  elif kernel == 'spectrum':
    X = Dataset["Xtr"]
    Y = Dataset["Ytr"]
    k = 4
    path_load_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
                            "Data/dataset_{}/SpecKernel_k{}.npz"
                            .format(dataset, k))
    K = np.load(path_load_kernel_mat)["Ktr"]
  else:
    raise ValueError("Kernel {} not implemented".format(kernel))
  # normalize
  K = normalize(K)

  #########################################################################
  #  simple run
  #########################################################################

  # perm = np.random.permutation(K.shape[0])
  # acc_train, acc_test = run(K, Y, cut, perm, tau)
  # print('\n acc_train ', acc_train)
  # print('\n acc_test', acc_test)

  #########################################################################
  #  cross validate
  #########################################################################

  taus = [1e-2, 1e-4, 1e-6, 1e-8]
  n_partitions = 4
  evaluate_taus(K, Y, cut, taus, n_partitions=4, kernel=kernel)

