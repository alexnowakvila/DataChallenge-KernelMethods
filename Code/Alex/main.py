import os
import numpy as np
from utils import *
from svm import *
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(0)  # set random seed
dataset = 0
solver = 'cvxopt'
# solver = 'mine'
path_data = ("/home/alexnowak/DataChallenge-KernelMethods/Data/")
Dataset = read_data(path_data, dataset=dataset)
X = Dataset["Xtr_mat50"].T
Y = Dataset["Ytr"]
# X_test = Dataset["Xte_mat50"].T

# pre-process data
permutation = np.random.permutation(X.shape[1])
X, Y = X[:, permutation], Y[permutation]
cut = 1700
X_train, X_test = X[:, :cut], X[:, cut:]
Y_train, Y_test = Y[:cut], Y[cut:]
# X_train = X_test = X
# Y_train = Y_test = Y
tau = 1e-6

# if data not centered add extra dimension
X_train = np.concatenate((X_train, np.ones((1, X_train.shape[1]))), axis=0)
X_test = np.concatenate((X_test, np.ones((1, X_test.shape[1]))), axis=0)
n = X_train.shape[1]
d = X_train.shape[0]

# barrier method parameters
t0 = 1.
mu = 3
tol = 1e-3
LS= True

#############################################################################
#  primal
#############################################################################

out = svm_solver(tau, X_train, Y_train, t0, mu, tol,
                 LS=LS, model='primal', solver=solver)
x_sol, xhist, fhist, fhist_b, w, acc_train = out
acc_test = compute_accuracy(X_test, Y_test, w)
print('acc_train', acc_train)
print('acc_test', acc_test)
print('PRIMAL OPTIMIZATION FINISHED')
if solver == 'mine':
  plt.semilogy(fhist)
  plt.semilogy(fhist_b, '--')
  plt.title('Primal $t_0$={}, $\mu$={}'.format(t0, mu))
  plt.xlabel('iterations')
  plt.show()
  print('PRIMAL RESULT: {:0.10f}'.format(fhist[-1]))
  primal_seq = fhist

#############################################################################
#  dual
#############################################################################

out = svm_solver(tau, X_train, Y_train, t0, mu, tol,
                 LS=LS, model='dual', solver=solver)
x_sol, xhist, fhist, fhist_b, w, acc_train = out
acc_test = compute_accuracy(X_test, Y_test, w)
print('acc_train', acc_train)
print('acc_test', acc_test)
print('DUAL OPTIMIZATION FINISHED')
if solver == 'mine':
  plt.semilogy(-1*fhist)
  plt.semilogy(-1*fhist_b, '--')
  plt.title('Dual $t_0$={}, $\mu$={}'.format(t0, mu))
  plt.xlabel('iterations')
  plt.show()
  print('DUAL RESULT: {:0.10f}'.format(-fhist[-1]))
  dual_seq = fhist



# #############################################################################
# #  Try different mu
# #############################################################################

# t0 = 1.
# mus = [3, 15, 50, 100]
# LS = True
# legend = []
# for mu in mus:
#   outp = svm_solver(tau, X_train, Y_train, t0, mu, tol, LS=LS, model='primal')
#   primal_seq = outp[2]
#   outd = svm_solver(tau, X_train, Y_train, t0, mu, tol, LS=LS, model='dual')
#   dual_seq = outd[2]
#   mn = min(primal_seq.shape[0], dual_seq.shape[0])
#   duality_gap = list(np.abs(primal_seq[:mn] + dual_seq[:mn]))
#   l, =plt.semilogy(duality_gap, label='$mu$={}'.format(mu))
#   legend.append(l)
# plt.legend(handles=legend)
# plt.xlabel('newton iterations')
# title = 'duality gaps'
# if LS:
#   title += ' with line search'
# plt.title(title)
# plt.show()

# #############################################################################
# #  Test different taus
# #############################################################################

# # test different values of tau
# taus = list(np.arange(0.001, 1.0, 0.01))
# accs = []
# for i, tau in enumerate(taus):
#   Q, p, A, b = transform_svm_primal(tau, X_train, Y_train)
#   # start from a strictly feasible point
#   x_0 = 2 * np.ones((d + n))
#   x_0[:d] = 0
#   x_sol, xhist,_ = barr_method(Q, p, A, b, x_0, mu, tol, t0=t0)
#   w = x_sol[:d]
#   acc = compute_accuracy(X_test, Y_test, w)
#   accs.append(acc)
# plt.plot(taus, accs)
# plt.xlabel('$tau$')
# plt.ylabel('test accuracy')
# plt.show()
