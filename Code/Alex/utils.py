import os
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm

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

    def predict(self, Xtr, Xte, alpha):
      K = np.dot(Xtr.T, Xte)  # has size (n1 x n2)
      y_pred = np.dot(K.T, alpha)
      return y_pred

###############################################################################
#  Spectrum Kernel
###############################################################################

class KernelSpectrum():
  def __init__(self, k=3, kernel_matrix=None):
    self.k = k
    if kernel_matrix is not None:
      self.K = K
    else:
      self.K = None

  def kernel_matrix(self, X):
    if self.K is None:
      X = self.preindex_strings(X).T
      n = X.shape[1]
      K = np.zeros((n, n))
      # diagonal
      for i in range(n):
        K[i, i] = self.kernel(X[:, i], X[:, i])
      # upper diagonal
      for i in tqdm(range(n), desc="Computing Spectrum Kernel Matrix"):
        for j in range(i+1, n):
          K[i, j] = self.kernel(X[:, i], X[:, j])
          K[j, i] = K[i, j]
      self.K = K
    return self.K

  def kernel(self, x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    d = {}
    K = 0
    for i in range(n1):
      if x1[i] in d:
        d[x1[i]] += 1
      else:
        d[x1[i]] = 1
    for j in range(n2):
      if x2[j] in d:
        K += d[x2[j]]
    return K

  def predict(self, Xtr, Xte, alpha):
    y_pred = []
    for xte in Xte:
      fx = sum(alpha * np.array([self.kernel(xtr, xte) for xtr in Xtr]))
      y_pred.append(fx)
    y_pred = np.array(y_pred)
    return y_pred

  def preindex_strings(self, X):
    X_num = []
    def char_to_num(c):
      if c == "A":
        return 0
      elif c == "C":
        return 1
      elif c == "G":
        return 2
      elif c == "T":
        return 3
      else:
        raise ValueError("Character {} not recognizable".format(c))
    def string_to_num(s):
      num = 0
      for i, c in enumerate(reversed(s)):
        num += (4**i) * char_to_num(c)
      return num
    for i, x in enumerate(X):
      n = len(x)
      x_num = []
      for j in range(n - self.k + 1):
        num = string_to_num(x[j:j+self.k])
        x_num.append(num)
      X_num.append(x_num)
    X_num = np.array(X_num)
    return X_num


if __name__ == "__main__":
  path_data = ("/home/alexnowak/DataChallenge-KernelMethods/"
               "Data/")
  Dataset0 = read_data(path_data, dataset=0)
  Dataset1 = read_data(path_data, dataset=1)
  Dataset2 = read_data(path_data, dataset=2)

  k = 4
  kernel = KernelSpectrum(k=k)
  dataset = 0
  path_save_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
                          "Data/dataset_{}/SpecKernel_k{}".format(dataset, k))
  Xtr = Dataset0["Xtr"]
  Xte = Dataset0["Xte"]
  Xtr_s = kernel.preindex_strings(Xtr).T
  Xte_s = kernel.preindex_strings(Xte).T
  # u = kernel.kernel(X_s[:, 0], X_s[:, 1])
  Ktr = kernel.kernel_matrix(Xtr_s)
  Kte = kernel.kernel_matrix(Xte_s)
  np.savez(path_save_kernel_mat, Ktr=Ktr, Kte=Kte)
  print("Saved!")
