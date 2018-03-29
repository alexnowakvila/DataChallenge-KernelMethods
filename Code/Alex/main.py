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
    acc_train, acc_test, _ = run_train(K, Y, cut, perm, tau)
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

#############################################################################
#  evaluate
#############################################################################

def evaluate_taus(K, Y, cut, taus, n_partitions=4):
  info, header = [], []
  for tau in taus:
    print("\n EVALUATING TAU = {}\n".format(tau))
    acc_tr, acc_te = cross_validate(K, Y, cut, tau, n_partitions=n_partitions)
    info.append([tau, acc_tr, acc_te])
  header = ["Tau", "Acc. Train", "Acc. Test"]
  print("\n", tabulate(info, header))

#############################################################################
#  train and test
#############################################################################


def run_train(K, Y, cut, perm, tau):
  # train test split
  K_train, K_test, Y_train, Y_test = kernel_train_test_split(K, Y, cut,
                                                             perm=perm)
  # solve
  svm_dual = kernel_SVM(tau, K_train, Y_train, squared=squared)
  alpha, acc_train = svm_dual.svm_solver()
  # test
  if cut < K.shape[0]:
    acc_test = svm_dual.compute_accuracy(K_test, Y_test)
  else:
    acc_test = -1
  return acc_train, acc_test, svm_dual

  ##########################################################################
  ##########################################################################
  ##########################################################################


if __name__ == "__main__":
  np.random.seed(1)  # set random seed
  dataset = 0
  main_path = "/home/alexnowak/DataChallenge-KernelMethods/"
  path_data = os.path.join(main_path, "Data/")
  Dataset = read_data(path_data, dataset=dataset)
  # regularization parameter
  tau = 1e-7
  cut = 1500
  test_size = 0.33
  squared = True
  submit = False
  # choose kernel
  X = Dataset["Xtr"]
  Y = Dataset["Ytr"]

  ##########################################################################
  #  mismatch kernel
  ##########################################################################

  # OPTIMAL CONFIGURATION: 
  ks = [1, 2, 4, 6, 8, 10]
  ms = [0, 0, 1, 1, 1, 1]
  weights = [.1, .1, .1, .1, .1, .1]  # weight of these kernels in the sum

  # FAST CHECK BY USING ONLY ONE KERNEL: 
  ks = [8]
  ms = [1]
  weights = [0.]

  for i, par in enumerate(zip(ks, ms, weights)):
    if submit:
      path_load_kernel_mat = (os.path.join(path_data,
                              "dataset_{}/MismKernel_k{}_m{}_all.npz"
                              .format(dataset, par[0], par[1])))
    else:
      path_load_kernel_mat = (os.path.join(path_data,
                              "dataset_{}/MismKernel_k{}_m{}.npz"
                              .format(dataset, par[0], par[1])))
    if i > 0:
      K = K + par[2] * np.load(path_load_kernel_mat)["Ktr"]
    else:
      K = par[2] * np.load(path_load_kernel_mat)["Ktr"]
  
  ##########################################################################
  #  shape kernel
  ##########################################################################

  weight_shape = 0.  # weight of this kernel in the sum
  if submit:
    path_load_kernel_mat = (os.path.join(path_data,
                            "dataset_{}/ShapeKernel_all.npz"
                            .format(dataset)))
  else:
    path_load_kernel_mat = (os.path.join(path_data,
                            "dataset_{}/ShapeKernel.npz"
                            .format(dataset)))
  Kshape = np.load(path_load_kernel_mat)["Ktr"]
  K = K + weight_shape * Kshape

  ##########################################################################
  #  substring kernel
  ##########################################################################

  weight_subs = 0.  # weight of this kernel in the sum
  if submit:
    path_load_kernel_mat = (os.path.join(path_data,
                            "dataset_{}/SubsKernel_all.npz"
                            .format(dataset)))
  else:
    path_load_kernel_mat = (os.path.join(path_data,
                            "dataset_{}/SubsKernel.npz"
                            .format(dataset)))

  Ksubs =  np.load(path_load_kernel_mat)["Ktr"]
  K = K +  weight_subs * Ksubs

  # normalize
  K = normalize(K)
  # if making a submission, split properly the data
  if submit:
    K_test = K[:2000, 2000:]
    K = K[:2000, :]
    K = K[:, :2000]
    
  #########################################################################
  #  simple run
  #########################################################################

  # perm = np.random.permutation(K.shape[0])
  # acc_train, acc_test, _ = run_train(K, Y, cut, perm, tau)
  # print('\n acc_train ', acc_train)
  # print('\n acc_test', acc_test)
  
  if submit:

    #########################################################################
    #  submit
    #########################################################################

    cut = 2000  # means that we do not validate in any other set
    tau = 0.4
    perm = None
    acc_train, acc_test, svm_dual = run_train(K, Y, cut, perm, tau)
    prediction = svm_dual.predict(K_test)
    path_submission = ("/home/alexnowak/DataChallenge-KernelMethods/"
                       "Data/dataset_{}/submission.npz".format(dataset))
    info = {"tau:": tau, "kernel": kernel, "k": ks, "m": ms, "w": weights}
    np.savez(path_submission, Y=prediction, info=info)
    print("Submission saved.")

  else:

    #########################################################################
    #  cross validate
    #########################################################################

    taus = [.04]
    n_partitions = 10
    evaluate_taus(K, Y, cut, taus, n_partitions=n_partitions)
