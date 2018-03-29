import os
import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pdb
from utils import *
import matplotlib.pyplot as plt

class QP_solver(object):
  def __init__(self, Q, p, A, b):
    self.Q = Q
    self.p = p
    self.A = A
    self.b = b

  def cvxopt_solver(self):
    Q, p, A, b = matrix(self.Q), matrix(self.p), matrix(self.A), matrix(self.b)
    solution = cvxopt.solvers.qp(Q, p, A, b)
    x_sol = np.array(solution['x'])[:,0]
    return x_sol


###############################################################################
#  Kernel-SVM
###############################################################################


class kernel_SVM(QP_solver):
  def __init__(self, lambd, K, y, squared=False):
    self.n = K.shape[0]
    self.y = y
    self.K = K
    self.alpha = -1
    if squared:
      Q, p, A, b = self.transform_squared_svm_dual(lambd, K, y)
    else:
      Q, p, A, b = self.transform_svm_dual(lambd, K, y)
    QP_solver.__init__(self, Q, p, A, b)

  ##############################################
  #  Hinge Loss
  ##############################################

  def transform_svm_dual(self, lambd, K, y):
    Q = 2 * K
    p = -2 * y.astype(float)
    A = np.concatenate((np.diag(y.astype(float)), -1*np.diag(y.astype(float))), axis=0)
    ones_n = np.ones((self.n))
    b = np.concatenate(((1/(2*lambd*self.n))* ones_n, np.zeros((self.n))), axis=0)
    return Q, p, A, b

  ##############################################
  #  Squared Hinge Loss
  ##############################################

  def transform_squared_svm_dual(self, lambd, K, y):
    Q = 2 * (K + self.n * lambd * np.eye(self.n))
    p = -2 * y.astype(float)
    A = -1 * np.diag(y.astype(float))
    b = np.zeros((self.n))
    return Q, p, A, b

  def svm_solver(self):
    alpha = self.cvxopt_solver()
    self.alpha = alpha  # update alpha
    acc = self.compute_accuracy(self.K, self.y)
    return alpha, acc

  def compute_accuracy(self, K, y):
    # K is a (ntr x nte) matrix
    y_pred = np.dot(np.expand_dims(self.alpha, 0), K).T
    y_pred = y_pred[:, 0]
    # y_pred = self.Kernel.predict(self.X, X, alpha)
    correct = ((y * y_pred) >= 0)
    acc = np.mean(correct)
    return acc

  def predict(self, K):
    # K is a (ntr x nte) matrix
    y_pred = np.dot(np.expand_dims(self.alpha, 0), K).T
    y_pred = y_pred[:, 0]
    prediction = (y_pred >= 0)
    return prediction


if __name__ == '__main__':
  path = '/home/alexnowak/Documents/MVA/ConvexOptimization/CO_HWK3/Data/'
  X, Y = read_data(path + 'bezdekIris.data')
  # test functions
  # Q=np.reshape(10, [1,1])
  # p=np.reshape(1, [1])
  # A=np.reshape(2, [1,1])
  # b=np.reshape(-1, [1])
  # x = np.reshape(-1, [1])
  # t = 1
  # mu = 1.5
  # tol = 1e-8
  # x_0 = x
  # ph = phi(x, t, Q, p, A, b)
  # gra = grad(x, t, Q, p, A, b)
  # hes = hess(x, t, Q, p, A, b)
  # f = lambda x: phi(x, t, Q, p, A, b)
  # g = lambda x: grad(x, t, Q, p, A, b)
  # h = lambda x: hess(x, t, Q, p, A, b)
  # xnew, gap = dampedNewtonStep(x, f, g, h)
  # x_sol, xhist,_ = barr_method(Q, p, A, b, x_0, mu, tol, t0=t, LS=True)
  # f_primal = (lambda x: 0.5*np.dot(np.dot(x.transpose(), Q), x) + 
  #             np.dot(p.transpose(), x))
  # hist_p = [f_primal(x) for x in xhist]
  # plt.plot(hist_p)
  # plt.show()

  # pre-process data
  np.random.seed(0)
  permutation = np.random.permutation(X.shape[1])
  X, Y = X[:, permutation], Y[permutation]
  X_train, X_test = X[:, :80], X[:, 80:]
  Y_train, Y_test = Y[:80], Y[80:]
  tau = 0.05
  # if data not centered add extra dimension
  X_train = np.concatenate((X_train, np.ones((1, X_train.shape[1]))), axis=0)
  X_test = np.concatenate((X_test, np.ones((1, X_test.shape[1]))), axis=0)
  n = X_train.shape[1]
  d = X_train.shape[0]

  # barrier method parameters
  t0 = 1.
  mu = 3
  tol = 1e-7
  LS= True

  #############################################################################
  #  primal
  #############################################################################
  
  out = svm_solver(tau, X_train, Y_train, t0, mu, tol, LS=LS, model='primal')
  x_sol, xhist, fhist, fhist_b, w, acc = out
  print('acc', acc)
  print('PRIMAL OPTIMIZATION FINISHED')
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

  out = svm_solver(tau, X_train, Y_train, t0, mu, tol, LS=LS, model='dual')
  x_sol, xhist, fhist, fhist_b, w, acc = out
  print('acc', acc)
  print('DUAL OPTIMIZATION FINISHED')
  plt.semilogy(-1*fhist)
  plt.semilogy(-1*fhist_b, '--')
  plt.title('Dual $t_0$={}, $\mu$={}'.format(t0, mu))
  plt.xlabel('iterations')
  plt.show()
  print('DUAL RESULT: {:0.10f}'.format(-fhist[-1]))
  dual_seq = fhist

  #############################################################################
  #  ex 3.4
  #############################################################################

  t0 = 1.
  mus = [3, 15, 50, 100]
  LS = True
  legend = []
  for mu in mus:
    outp = svm_solver(tau, X_train, Y_train, t0, mu, tol, LS=LS, model='primal')
    primal_seq = outp[2]
    outd = svm_solver(tau, X_train, Y_train, t0, mu, tol, LS=LS, model='dual')
    dual_seq = outd[2]
    mn = min(primal_seq.shape[0], dual_seq.shape[0])
    duality_gap = list(np.abs(primal_seq[:mn] + dual_seq[:mn]))
    l, =plt.semilogy(duality_gap, label='$mu$={}'.format(mu))
    legend.append(l)
  plt.legend(handles=legend)
  plt.xlabel('newton iterations')
  title = 'duality gaps'
  if LS:
    title += ' with line search'
  plt.title(title)
  plt.show()

  #############################################################################
  #  Test different taus
  #############################################################################

  # test different values of tau
  taus = list(np.arange(0.001, 1.0, 0.01))
  accs = []
  for i, tau in enumerate(taus):
    Q, p, A, b = transform_svm_primal(tau, X_train, Y_train)
    # start from a strictly feasible point
    x_0 = 2 * np.ones((d + n))
    x_0[:d] = 0
    x_sol, xhist,_ = barr_method(Q, p, A, b, x_0, mu, tol, t0=t0)
    w = x_sol[:d]
    acc = compute_accuracy(X_test, Y_test, w)
    accs.append(acc)
  plt.plot(taus, accs)
  plt.xlabel('$tau$')
  plt.ylabel('test accuracy')
  plt.show()