import numpy as np
import sys
sys.path.insert(0, '/home/alexnowak/DataChallenge-KernelMethods/Code/Alex/')
sys.path.insert(0, '/home/alexnowak/DataChallenge-KernelMethods/Code/Adrien/')

from utils import *
from svm2 import *

####################################
#  Data
####################################

dataset = 0
folder = '/home/alexnowak/DataChallenge-KernelMethods/Data/'
data = read_data(folder, dataset=dataset)
print(data.keys())

####################################
#  Substring Kernel
####################################

def substring_kernel(x, y, gamma, k):
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

substring_kernel('lpax', 'mnih', 0.8, 2)

####################################
#  Approx Substring Kernel
####################################

# nb_test = 100
# x_test = data['Xtr'][0:nb_test]
# alphabet = ['A', 'C', 'T', 'G']
# k = 4
# l = 0.8
# substrings = []
# for i1 in alphabet:
#   for i2 in alphabet:
#     for i3 in alphabet:
#       for i4 in alphabet: 
#         if i1+i2+i3+i4 not in substrings:
#           substrings.append(i1+i2+i3+i4)

# len(substrings)
# features = np.zeros((nb_test, len(alphabet)**k))
# for i in range(nb_test):
#   for j in tqdm(range(len(substrings))):
#     features[i, j] = substring_kernel(x_test[i], substrings[j], l, k)

# K_approx = np.zeros((nb_test, nb_test))
# K_real = np.zeros((nb_test, nb_test))
# constant = substring_kernel(substrings[0], substrings[0], l, k)

# for i in range(nb_test):
#   for j in range(i+1, nb_test):
#     K_approx[i, i] = features[i, :].T.dot(features[i, :]) / constant
#     K_approx[i, j] = features[i, :].T.dot(features[j, :]) / constant
#     K_approx[j, i] = K_approx[i, j]

# for i in range(nb_test):
#   for j in tqdm(range(i+1, nb_test)):
#     K_real[i, i] = substring_kernel(x_test[i], x_test[i], l, k)
#     K_real[i, j] = substring_kernel(x_test[i], x_test[j], l, k)
#     K_real[j, i] = K_real[i, j]

# def alignment(K_1, K_2):
#   if K_1.shape != K_2.shape:
#     return False
#   return np.sum(K_1*K_2) / np.sqrt(np.sum(K_1*K_1) * np.sum(K_2*K_2))

# alignment(K_approx, K_real)
# K_base = np.zeros((len(substrings), len(substrings)))

# for i in tqdm(range(len(substrings))):
#   for j in range(i+1, len(substrings)):
#     K_base[i, i] = substring_kernel(substrings[i], substrings[i], l, k)
#     K_base[i, j] = substring_kernel(substrings[i], substrings[j], l, k)
#     K_base[j, i] = K_base[i, j]

full_alphabet = (['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                  'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                  'y', 'z'])

full_substrings = []
for x in full_alphabet:
  for y in full_alphabet:
    if x+y not in full_substrings:
        full_substrings.append(x+y)

def substring_explicit(subs, lamb, x, y):
  K = 0
  for t in tqdm(range(len(subs))):
    u = subs[t]
    l_x = []
    l_y = []
    for i1 in range(len(x)):
      if x[i1] == u[0]:
        for i2 in range(i1+1, len(x)):
          if x[i2] == u[1]:
            l_x.append(i2-i1+1)
            """
            for i3 in range(i2+1, len(x)):
              if x[i3] == u[2]:
                for i4 in range(i3+1, len(x)):
                  if x[i4] == u[3]:
                    l_x.append(i4-i1+1)
              """
    for j1 in range(len(y)):
      if y[j1] == u[0]:
        for j2 in range(j1+1, len(y)):
          if y[j2] == u[1]:
            l_y.append(j2-j1+1)
            """
            for j3 in range(j2+1, len(y)):
                if y[j3] == u[2]:
                    for j4 in range(j3+1, len(y)):
                        if y[j4] == u[3]:
                            l_y.append(j4-j1+1)
            """                                 
    for i in l_x:
      for j in l_y:
        K += lamb**(i+j)
    #print(u, K, l_x, l_y)  
  return K


# substring_explicit(substrings, l, 'ATCCTGAGCTCCACTACTA', 'ATCCTGAGCTCCACTACTA')
# substring_kernel('ATCCTGAGCTCCACTACTG', 'ATCCTGAGCTCCACTACTG', l, 4)
# features[0, :].T.dot(features[0, :]) / constant
# substring_explicit(substrings, l, 'ATCC', 'ATCC')
# substring_kernel('ATCC', 'ATCC', l, 4)

def prod_log(a, b):
  res = a + b
  return res

def sum_log(a, b):
  res = np.log(a) + np.log(1 + np.exp(b - a))
  return res

def power(a, b):
  res =  b * np.log(a)
  return res

def K(s, t, gamma, N):
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
  return grid_K[S-1, T-1]

app = K('catklbfgiezbkjjgdcjeaehzjazzaahdbevkjhugsdvyrusdbuc',
        'azzaahdbevkjhugsdvyrusdbuccatklbfgiezbkjjgdcjeaehzj', 0.8, 2)

exact = substring_explicit(full_substrings, 0.8,
                   'catklbfgiezbkjjgdcjeaehzjazzaahdbevkjhugsdvyrusdbuc',
                   'azzaahdbevkjhugsdvyrusdbuccatklbfgiezbkjjgdcjeaehzj')

print(app, exact)

########################################################
#  main
########################################################

dataset = 0
test_size = 0.25
tau = 1e-1
k = 4
gamma = 0.5
from sklearn.model_selection import train_test_split
X = data['Xtr']
Y = data['Ytr']
from dev import *
Kernel = Kernel_Substring(k, gamma)
K = Kernel.kernel_matrix(X)
cut = 1500
Ktr = K[:cut,:]
Ktr = Ktr[:, :cut]
Kte = K[:cut,cut:]
svm_dual = kernel_SVM(tau, Ktr, Y_train, squared=True)
x_sol, alpha, acc_train = svm_dual.svm_solver()
acc_test = svm_dual.compute_accuracy(Kte, Y_test)
print('acc_train', acc_train)
print('acc_test', acc_test)
print('DUAL OPTIMIZATION FINISHED')
len(X_train[0])