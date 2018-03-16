import os
from subprocess import Popen, PIPE, STDOUT, call
import pandas as pd
import numpy as np
import pdb

Datasets = [0, 1, 2]
k = 6
m = 2
for dataset in Datasets:
  print("\nCREATING DATASET {} FOR K = {} AND M = {}\n".format(dataset, k, m))
  trad = {"A": 0, "C": 1, "G": 2, "T": 3}
  data_path = ("/home/alexnowak/DataChallenge-KernelMethods/"
               "Data/dataset_{}/".format(dataset))
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
  path_load = "Kernel-k8-m2.txt"
  Ktr = pd.read_csv(path_load, sep=' ', header=None)
  Ktr = Ktr.as_matrix()[:,:-1]
  np.savez(path_save_kernel_mat, Ktr=Ktr)
  print("Saved!")
pdb.set_trace()

