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

  ##############################################
  #  Efficient implementation from Farhan et al.
  ##############################################

  def compute_nij(self, i, j, d):
    nij = 0
    for t in range(i+j-d/2):
      a1 = comb(2*d-i-j+2*t, d-(i-t))
      a2 = comb(d, i+j-2*t-d)
      a3 = np.power(self.l-2, i+j-2*t-d)
      a4 = comb(k-d, t)
      a5 = np.power(self.l-1, t)
      nij += a1 * a2 * a3 * a4 * a5
    return nij

  def compute_I(self):
    I = []
    for d in range(self.k):
      # compute the size of the intersection when
      Nd = 0
      for i in range(self.m):
        for j in range(self.m):
          Nd += self.compute_nij(i, j, d)
      I.append(Nd)
    return I

  def sort_enumerate(self, x1_kmers, x2_kmers, theta):
    """ This routine first orders the k-mers of x1 and x2 lexicographically
    and then enumerates the pairs in Sx x Sy that coincide at the indices
    given by theta with a linear pass """
    nX = len(x1_kmers)
    # first extract the coordinates corresponding to theta and sort
    x1_theta = [x1_kmers[r, theta] for r in range(nX)].sort()
    x2_theta = [x2_kmers[r, theta] for r in range(nX)].sort()
    # do a linear scan to find the k-mers that coincide in theta
    f_theta = 0
    t1, t2 = 0, 0
    while (t1 < nX) and (t2 < nX):
      if x1_theta[t1] < x2_theta[t2]:
        t1 += 1
      elif x1_theta[t1] > x2_theta[t2]:
        t2 += 1
      else:
        # they are equal
        f_theta += 1
    return f_theta

  def online_variance(self, muF, varF, f_theta, it):
    varF = (varF * (it-1) + f_theta**2) / it - muF**2
    return varF

  def kernel_efficient(self, x1, x2, sigma=0.5, B=200):
    n = len(x1)
    assert n == len(x2)
    I = self.compute_I()
    M = []
    # construct Sx and Sy
    x1_kmers = [''.join(x1[r:r+self.k]) for r in range(n-self.k)]
    x2_kmers = [''.join(x2[r:r+self.k]) for r in range(n-self.k)]
    for i in range(self.t):
      muF = 0
      it = 1
      varF = 1e6
      while (varF > sigma**2) and (it < B):
        theta = np.random.permutation(self.k)[:(k-i)]
        f_theta = self.sort_enumerate(x1_kmers, x2_kmers, theta)
        muF = (muF * (it-1) + f_theta) / it
        varF = self.online_variance(muF, varF, f_theta, it)
        it += 1
      F_i = muF * comb(self.k, self.k-i)
      M.append(F_i)
      for j in range(i-1):
        M[i] = M[i] - comb(self.k-j, self.k-i) * M[j]
    # sum-product
    K = sum([M[r]*I[r] for r in range(self.t)])
    return K


if __name__ == "__main__":
  path_data = ("/home/alexnowak/DataChallenge-KernelMethods/"
               "Data/")
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
  #  Create Spectrum Kernel
  #############################################################################

  # k = 4
  # kernel = KernelSpectrum(k=k)
  # dataset = 0
  # path_save_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
  #                         "Data/dataset_{}/SpecKernel_k{}".format(dataset, k))
  # Xtr = Dataset0["Xtr"]
  # Xte = Dataset0["Xte"]
  # # u = kernel.kernel(X_s[:, 0], X_s[:, 1])
  # Ktr = kernel.kernel_matrix(Xtr)
  # Kte = kernel.kernel_matrix(Xte)
  # N = Ktr.shape[0]
  # # IU = np.eye(N) - (1/N)*np.ones((N,N))
  # # Ktr = np.dot(np.dot(IU, Ktr), IU)  # center
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
  # kernel2 = KernelSpectrum(k=k)
  # x1, x2 = Xtr[0][:10], Xtr[1][:10]
  # x1 = ['C', 'G', 'G']
  # x2 = ['C', 'A', 'T']
  # a = kernel.kernel(x1, x2)
  # print(a)
  # b = kernel2.kernel(x1, x2)
  # print(b)
  # kernel.compute_I()
  pdb.set_trace()
  Ktr = kernel.kernel_matrix(Xtr)
  Kte = kernel.kernel_matrix(Xte)
  np.savez(path_save_kernel_mat, Ktr=Ktr, Kte=Kte)
  print("Saved!")
  pdb.set_trace()