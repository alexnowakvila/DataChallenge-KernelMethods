import os
from subprocess import Popen, PIPE, STDOUT, call
import pandas as pd
import numpy as np
import pdb

Datasets = [0, 1, 2]
# run = "training"
run = "submission"
k = 2
m = 1
# b = 400
# sigma = 0.2
for k in [4, 6, 8, 10]:
  for dataset in Datasets:
    print("\nCREATING {} DATASET {} FOR K = {} AND M = {}\n"
          .format(run, dataset, k, m))
    trad = {"A": 0, "C": 1, "G": 2, "T": 3}
    data_path = ("/home/alexnowak/DataChallenge-KernelMethods/"
                 "Data/dataset_{}/".format(dataset))
    if run == "training":
      #############################################################################
      #  Training Kernel
      #############################################################################

      path = os.path.join(data_path, "Xtr{}.csv".format(dataset))
      with open(path, "r") as file:
        X = []
        for f in file:
          X.append(list(f[:-2]))
      for i, x in enumerate(X):
        X[i] = [trad[c] for c in x]
      X = np.array(X)
      D = pd.DataFrame(X)
      path_save = ("/home/alexnowak/DataChallenge-KernelMethods/"
                   "Code/Alex/mismatch/Datasets/dataset_{}_Xtr.txt"
                   .format(dataset))
      D.to_csv(path_save, sep=' ', header=None, index=None)
      src = "java Main"
      n = X.shape[0]
      cmd = ' '.join([src, path_save, str(n), str(k), str(m), str(4)])
      # p = Popen(cmd, shell=True)
      call(cmd, shell=True)
      path_save_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
                              "Data/dataset_{}/MismKernel_k{}_m{}"
                              .format(dataset, k, m))
      path_load = "Kernel-k{}-m{}.txt".format(k, m)
      Ktr = pd.read_csv(path_load, sep=' ', header=None)
      Ktr = Ktr.as_matrix()[:,:-1]
      np.savez(path_save_kernel_mat, Ktr=Ktr)
      print("Saved!")
    elif run == "submission":
      #############################################################################
      #  Submission Kernel
      #############################################################################

      X = []
      path = os.path.join(data_path, "Xtr{}.csv".format(dataset))
      with open(path, "r") as file:
        for f in file:
          X.append(list(f[:-2]))
      path = os.path.join(data_path, "Xte{}.csv".format(dataset))
      with open(path, "r") as file:
        for f in file:
          X.append(list(f[:-2]))
      for i, x in enumerate(X):
        X[i] = [trad[c] for c in x]
      X = np.array(X)
      D = pd.DataFrame(X)
      path_save = ("/home/alexnowak/DataChallenge-KernelMethods/"
                   "Code/Alex/mismatch/Datasets/dataset_{}_all.txt"
                   .format(dataset))
      D.to_csv(path_save, sep=' ', header=None, index=None)
      src = "java Main"
      n = X.shape[0]
      cmd = ' '.join([src, path_save, str(n), str(k), str(m), str(4)])
      # p = Popen(cmd, shell=True)
      call(cmd, shell=True)
      path_save_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
                              "Data/dataset_{}/MismKernel_k{}_m{}_all"
                              .format(dataset, k, m))
      path_load = "Kernel-k{}-m{}.txt".format(k, m)
      Ktr = pd.read_csv(path_load, sep=' ', header=None)
      Ktr = Ktr.as_matrix()[:,:-1]
      np.savez(path_save_kernel_mat, Ktr=Ktr)
      print("Saved!")


