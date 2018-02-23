import os
import pdb
import numpy as np
import pandas as pd

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

def compute_accuracy(X, y, w):
  X = X.transpose()
  acc = 0
  for i in range(X.shape[0]):
    if np.dot(X[i], w) >= 0 and y[i] == 1:
      acc += 1
    elif np.dot(X[i], w) < 0 and y[i] == -1:
      acc += 1
    else:
      pass
  acc /= X.shape[0]
  return acc


if __name__ == "__main__":
  path_data = ("/home/alexnowak/DataChallenge-KernelMethods/"
               "Data/")
  Dataset0 = read_data(path_data, dataset=0)
  Dataset1 = read_data(path_data, dataset=1)
  Dataset2 = read_data(path_data, dataset=2)
