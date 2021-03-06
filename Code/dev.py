import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/alexnowak/DataChallenge-KernelMethods/Code/')

from utils import *
from svm import *


class Kernel_Substring():

  def __init__(self, k, gamma, kernel_matrix=None):

    self.k = k
    self.gamma = gamma

    if kernel_matrix is not None:
      self.K = K
    else:
      self.K = None

  def KK(self, s, t, gamma, N):
    if N < 2:
      print('works only if N > 1')
      return False
    S = len(s)
    T = len(t)
    grid = np.ones((S, T, N), dtype=np.float64)
    grid_K = np.zeros((S, T), dtype=np.float64)
    #kernel = 0
    for p in range(1, N):
      for i in range(S):
        for j in range(T):   
          if min(i+1, j+1) < p:
            grid[i, j, p] = 0
          elif p == 1 and i == 0:
            sum_gamma = 0
            for n, a in enumerate(t[0:j+1]):
              if a == s[0]:
                sum_gamma += gamma**(j+1-n)
            grid[i, j, p] = gamma * sum_gamma
          else:
            sum_gamma = 0
            for n, a in enumerate(t[0:j+1]):
              if a == s[i]:
                sum_gamma += grid[i-1, n-1, p-1] * (gamma**(j-n+2))
            grid[i, j, p] = gamma * grid[i-1, j, p] + sum_gamma              
    """                    
    for i in range(N-1, S):
      sum_K = 0
      for n, a in enumerate(t):
        sum_K += grid[i-1, n-1, N-1]  
      kernel = kernel + sum_K * (gamma**2) 
      print(kernel)
    """     
    for i in range(S):
      for j in range(T):              
        if min(i+1, j+1) < N:
          continue
        else:
          sum_gamma = 0
          for n, a in enumerate(t[0:j+1]):
            if a == s[i]:
              sum_gamma += grid[i-1, n-1, N-1] * (gamma**2)
          grid_K[i, j] = grid_K[i-1, j] + sum_gamma
    print('here')
    return grid_K[S-1, T-1]

  def substring_kernel(self, x, y, gamma, k):
    """
    x, y: input strings
    gamma: real value
    k: length of substring
    
    compute substring kernel distance between x and y
    """
    N = len(x)
    M = len(y)
    grid_B = np.zeros((M, N, k+1))
    grid_B[:, :, 0] += np.ones((M, N))
    grid_K = np.zeros((M, N, k+1))
    
    for p in range(1, k+1):
        for i in range(M):
            for j in range(N):
                
                if min(i+1, j+1) < p:
                    continue
                
                sum_B = 0
                sum_K = 0
                
                if p == 1 and j == 0:
                    
                    for n, a in enumerate(y[0:i+1]):
                        
                        if a == x[0]:
                            
                            sum_B += gamma ** (i+1-n)
                            sum_K += gamma ** 2
                    
                    grid_B[i, 0, 1] = gamma * sum_B
                    grid_K[i, 0, 1] = sum_K
                 
                else:
                    
                    for n in range(1, i+1):
                        
                        if y[n] == x[j]:
                            
                            sum_B += grid_B[n-1, j-1, p-1] * (gamma ** (i-n+2))
                            sum_K += grid_B[n-1, j-1, p-1]
                    
                    grid_B[i, j, p] = gamma * grid_B[i, j-1, p] + sum_B
                    grid_K[i, j, p] = grid_K[i, j-1, p] + (gamma ** 2) * sum_K
    return grid_K[M-1, N-1, k]

  def kernel_matrix(self, X):

    if self.K is None:

      n = len(X)
      K = np.zeros((n, n))
      
      for i in tqdm(range(n), desc="Computing Spectrum Kernel Matrix"):

        # diagonal
        # K[i, i] = self.substring_kernel(X[i], X[i], self.gamma, self.k)
        K[i, i] = self.KK(X[i], X[i], self.gamma, self.k)

        # upper diagonal
        for j in range(i+1, n):
            # K[i, j] = self.substring_kernel(X[i], X[j], self.gamma, self.k)
            K[i, j] = self.KK(X[i], X[j], self.gamma, self.k)
            K[j, i] = K[i, j]

      self.K = K

    return self.K

  def predict(self, Xtr, Xte, alpha):

    y_pred = []

    for xte in Xte:
      fx = np.sum(alpha * np.array([self.substring_kernel(xtr, xte, self.gamma, self.k) for xtr in Xtr]))
      y_pred.append(fx)

    y_pred = np.array(y_pred)

    return y_pred


if __name__ == "__main__":
  path_data = ("/home/alexnowak/DataChallenge-KernelMethods/"
               "Data/")
  Dataset0 = read_data(path_data, dataset=0)
  Dataset1 = read_data(path_data, dataset=1)
  Dataset2 = read_data(path_data, dataset=2)

  #############################################################################
  #  Create Substring Kernel
  #############################################################################

  dataset = 0
  k = 1
  gamma = 0.8
  path_save_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
                          "Data/dataset_{}/SubsKernel".format(dataset))
  Xtr = Dataset0["Xtr_mat50"][:10]
  Xte = Dataset0["Xtr_mat50"][:10]
  kernel = Kernel_Substring(k, gamma)
  # u = kernel.kernel(X_s[:, 0], X_s[:, 1])
  Ktr = kernel.kernel_matrix(Xtr)
  Kte = kernel.kernel_matrix(Xte)
  np.savez(path_save_kernel_mat, Ktr=Ktr, Kte=Kte)
  print("Saved!")
  pdb.set_trace()