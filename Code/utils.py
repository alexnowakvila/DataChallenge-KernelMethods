import os
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import comb

n_datasets = 3

def read_data(path_data, dataset=0):
  Dataset = {}
  path_data = os.path.join(path_data, "dataset_{}/".format(dataset))
  # read matrices
  path = os.path.join(path_data, "Xtr{}_mat50.csv".format(dataset))
  with open(path, "r") as file:
    X = []
    for f in file:
      X.append(np.fromstring(f, dtype=float, sep=" "))
    X = np.array(X)
    Dataset["Xtr_mat50"] = X
  path = os.path.join(path_data, "Xte{}_mat50.csv".format(dataset))
  with open(path, "r") as file:
    X = []
    for f in file:
      X.append(np.fromstring(f, dtype=float, sep=" "))
    X = np.array(X)
    Dataset["Xte_mat50"] = X
  # read dna sequences
  path = os.path.join(path_data, "Xtr{}.csv".format(dataset))
  with open(path, "r") as file:
    X = []
    for f in file:
      X.append(list(f[:-2]))
    Dataset["Xtr"] = X
  path = os.path.join(path_data, "Xte{}.csv".format(dataset))
  with open(path, "r") as file:
    X = []
    for f in file:
      X.append(list(f[:-2]))
    Dataset["Xte"] = X
  path = os.path.join(path_data, "Ytr{}.csv".format(dataset))
  with open(path, "r") as file:
    X = []
    first = True
    for f in file:
      if not first:
        x = np.fromstring(f, dtype=int, sep=",")[1]
        if x == 0: x = -1
        X.append(x)
      first = False
    X = np.array(X)
    Dataset["Ytr"] = X
  return Dataset

def normalize(K):
  n = K.shape[0]
  # center
  # IU = np.eye(n) - (1/n)*np.ones((n,n))
  # K = np.dot(np.dot(IU, K), IU)
  # normalize it
  # k = np.expand_dims(1./np.sqrt(np.diag(K)), 1)
  # kkt = np.dot(k, k.T)
  # K = K * kkt
  return K

def kernel_train_test_split(K, Y, cut, perm=None):
  if perm is not None:
    # permute data
    K = K[perm]
    K = K[:, perm]
    Y = Y[perm]
  # cut data
  K_train = K[:cut, :cut]
  K_test = K[:cut, cut:]
  Y_train, Y_test = Y[:cut], Y[cut:]
  return K_train, K_test, Y_train, Y_test

###############################################################################
#  Linear Kernel
###############################################################################

class KernelLinear():
    def __init__(self, dim=10):
      self.d = dim

    def kernel_matrix(self, X):
      # X has size (d x n)
      K = np.dot(X.T, X)
      return K

    def kernel(self, x, y):
      Kxy = np.dot(x.T, y)
      return Kxy

###############################################################################
#  Mismatch Kernel
###############################################################################

class KernelMismatch():
  def __init__(self, k=4, m=1, kernel_matrix=None):
    self.k = k
    self.m = m
    self.t = min(2*m, k)
    self.trad = {"A": 0, "C": 1, "G": 2, "T": 3}
    self.l = len(self.trad)
    if kernel_matrix is not None:
      self.K = K
    else:
      self.K = None

  def update_dict(self, dic, x, d, c):
    new_dic = {}
    for ptr in dic.keys():
      if self.trad[x[ptr + d]] == c:
        new_dic[ptr] = dic[ptr]
      elif dic[ptr] < self.m:
        new_dic[ptr] = dic[ptr] + 1
      else:
        pass
    return new_dic

  def mismatch_tree(self, K, x1, x2, dict1, dict2, n, d=0):
    if d == k:
      K += len(dict1.keys()) * len(dict2.keys())
      # pdb.set_trace()
      return K
    else:
      for c in range(len(self.trad)):
        if len(dict1) > 0 and len(dict2) > 0:
          new_dict1 = self.update_dict(dict1, x1, d, c)
          new_dict2 = self.update_dict(dict2, x2, d, c)
          K = self.mismatch_tree(K, x1, x2, new_dict1, new_dict2, n, d+1)
      return K

  def kernel(self, x1, x2):
    n = len(x1)
    assert n == len(x2)
    dict1, dict2 = {}, {}
    for i in range(n - self.k + 1):
      dict1[i], dict2[i] = 0, 0
    K = 0
    K = self.mismatch_tree(K, x1, x2, dict1, dict2, n, d=0)
    return K

  def kernel_matrix(self, X):
    if self.K is None:
      n = len(X)
      K = np.zeros((n, n))
      # diagonal
      for i in range(n):
        K[i, i] = self.kernel(X[i], X[i])
      # upper diagonal
      for i in tqdm(range(n), desc="Computing Spectrum Kernel Matrix"):
        for j in range(i+1, n):
          K[i, j] = self.kernel(X[i], X[j])
          K[j, i] = K[i, j]
      self.K = K
    return self.K


if __name__ == "__main__":
  main_path = "/home/alexnowak/DataChallenge-KernelMethods/"
  path_data = os.path.join(main_path, "Data/")
  Dataset0 = read_data(path_data, dataset=0)
  Dataset1 = read_data(path_data, dataset=1)
  Dataset2 = read_data(path_data, dataset=2)

  #############################################################################
  #  Create Linear Kernel
  #############################################################################

  # dataset = 0
  # path_save_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
  #                         "Data/dataset_{}/LinKernel".format(dataset))
  # Xtr = Dataset0["Xtr_mat50"]
  # Xte = Dataset0["Xtr_mat50"]
  # kernel = KernelLinear(dim=Xtr.shape[0])
  # # u = kernel.kernel(X_s[:, 0], X_s[:, 1])
  # Ktr = kernel.kernel_matrix(Xtr)
  # Kte = kernel.kernel_matrix(Xte)
  # np.savez(path_save_kernel_mat, Ktr=Ktr, Kte=Kte)
  # print("Saved!")
  # pdb.set_trace()

  #############################################################################
  #  Create Mismatch Kernel
  #############################################################################
  
  k = 1
  m = 0
  kernel = KernelMismatch(k=k, m=m)
  dataset = 0
  path_save_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
                          "Data/dataset_{}/MismKernel_k{}_m{}"
                          .format(dataset, k, m))
  Xtr = Dataset0["Xtr"]
  Xte = Dataset0["Xte"]
  
  Ktr = kernel.kernel_matrix(Xtr)
  Kte = kernel.kernel_matrix(Xte)

  np.savez(path_save_kernel_mat, Ktr=Ktr, Kte=Kte)
  print("Saved!")
