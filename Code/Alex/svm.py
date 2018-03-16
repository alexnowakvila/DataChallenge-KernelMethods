import os
import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pdb
from utils import *
import matplotlib.pyplot as plt

class QP_solver(object):
  def __init__(self, Q, p, A, b, x_0, mu, tol, t0=1., LS=False):
    self.Q = Q
    self.p = p
    self.A = A
    self.b = b
    self.x_0 = x_0
    self.mu = mu
    self.tol = tol
    self.t0 = t0
    self.LS = LS

  def phi(self, x, t):
    phi_1 = (t*(0.5*np.dot(np.dot(x.transpose(), self.Q), x)
             + np.dot(self.p.transpose(), x)))
    dif = self.b - np.dot(self.A, x)
    barrier = -1 * np.sum(np.log(dif))
    phi = phi_1 + barrier
    return phi

  def grad(self, x, t):
    grad_phi_1 = t*(np.dot(self.Q,x) + self.p)
    dif = self.b - np.dot(self.A, x)
    grad_barrier = np.sum(self.A / np.expand_dims(dif, 1), axis=0)
    return grad_phi_1 + grad_barrier

  def hess(self, x, t):
    hess_phi_1 = t * self.Q
    dif = self.b - np.dot(self.A, x)
    A_1 = self.A / np.expand_dims(dif, 1)
    hess_barrier = np.dot(A_1.transpose(), A_1)
    return hess_phi_1 + hess_barrier

  def dampedNewtonStep(self, x, f, g, h):
    fx, gx, hx = f(x), g(x), h(x)
    hx_inv = np.linalg.inv(hx)
    lambd = np.sqrt(np.dot(np.dot(gx.transpose(), hx_inv), gx))
    lr = 1./((1 + lambd))
    # F1, F2 = [],[]
    # T = np.arange(-1, 1, 0.001)
    # for t in list(T):
    #   F1.append(f(x - t*np.dot(hx_inv, gx)))
    #   F2.append( (f(x) - t*lambd**2) )
    # plt.plot(list(T), F1, 'r')
    # plt.plot(list(T), F2, 'b')
    # plt.show()
    xnew = x - lr*np.dot(hx_inv, gx)
    gap = 0.5*lambd*lambd
    return xnew, gap

  def dampedNewton(self, x0, f, g, h):
    max_tol = (3 - np.sqrt(5))/2
    try:
      assert self.tol < max_tol
    except:
      raise ValueError('tolerance must be smaller than {}'.format(max_tol))
    # compute first step
    xnew, gap = self.dampedNewtonStep(x0, f, g, h)
    xhist = [xnew]
    while gap > self.tol:
      if self.LS:
        xnew, gap = self.LSNewtonStep(xnew, f, g, h)
      else:
        xnew, gap = self.dampedNewtonStep(xnew, f, g, h)
      xhist.append(xnew)
    xstar = xnew
    return xstar, xhist

  def linesearch(self, x, deltax, gx, f, g, alpha=0.4, beta=0.9):
    gx = g(x)
    d = np.dot(gx.transpose(), deltax)
    t = 1
    i = 0
    while np.isnan(f(x + t*deltax)) or f(x + t*deltax) > (f(x) + alpha*t*d):
      t *= beta
      i += 1
    return t

  def LSNewtonStep(self, x, f, g, h):
    fx, gx, hx = f(x), g(x), h(x)
    hx_inv = np.linalg.inv(hx)
    lambd = np.sqrt(np.dot(np.dot(gx.transpose(), hx_inv), gx))
    deltax = -1 * np.dot(hx_inv, gx)
    lr = self.linesearch(x, deltax, gx, f, g)
    xnew = x + lr*deltax
    gap = 0.5*lambd*lambd
    return xnew, gap

  def barr_method(self):
    t = self.t0
    m = self.A.shape[0]
    x0 = self.x_0
    xhist = [x0]
    barriers = []
    while m/t > self.tol:
      f = lambda x: self.phi(x, t)
      g = lambda x: self.grad(x, t)
      h = lambda x: self.hess(x, t)
      xstar, xhist_dn = self.dampedNewton(x0, f, g, h)
      xhist.extend(xhist_dn)
      barriers.extend([x0 for i in xhist_dn])
      x0 = xstar
      t *= self.mu
    x_sol = xstar
    return x_sol, xhist, barriers

  def cvxopt_solver(self):
    Q, p, A, b = matrix(self.Q), matrix(self.p), matrix(self.A), matrix(self.b)
    solution = cvxopt.solvers.qp(Q, p, A, b)
    x_sol = np.array(solution['x'])[:,0]
    return x_sol


###############################################################################
#  Euclidean-SVM
###############################################################################


class Euclidean_SVM(QP_solver):
  def __init__(self, tau, X, y, dual=True):
    self.n = X.shape[1]
    self.d = X.shape[0]
    self.X = X
    self.y = y
    self.dual = dual
    mu = 3.
    tol = 1e-1
    LS = True
    if dual:
      Q, p, A, b = self.transform_svm_dual(tau, X, y)
      x_0 = (1/(2*tau*self.n))*np.ones((self.n))
    elif not dual:
      Q, p, A, b = self.transform_svm_primal(tau, X, y)
      x_0 = 10 * np.ones((self.d + self.n))
      x_0[:self.d] = 0
    QP_solver.__init__(self, Q, p, A, b, x_0, mu, tol, t0=1.2, LS=True)

  def transform_svm_primal(self, tau, X, y):
    X_y = X * np.expand_dims(y, 0)  # d x n
    A_1 = np.concatenate((X_y.transpose(), np.eye(self.n)), axis=1)
    A_2 = np.concatenate((np.zeros((self.n, self.d)), np.eye(self.n)), axis=1)
    A = -1 * np.concatenate((A_1, A_2), axis=0)
    b = -1 * np.concatenate((np.ones((self.n)), np.zeros((self.n))), axis=0)
    Q_1 = np.concatenate((np.eye(self.d), np.zeros((self.d, self.n))), axis=1)
    Q_2 = np.zeros((self.n, self.d + self.n))
    Q = np.concatenate((Q_1, Q_2), axis=0)
    p = (np.concatenate((np.zeros((self.d)), (1/(tau*self.n)) *
         np.ones((self.n))), axis=0))
    return Q, p, A, b

  def transform_svm_dual(self, tau, X, y):
    self.n = y.shape[0]
    X_y = X * np.expand_dims(y, 0)  # d x n
    Q = np.dot(X_y.transpose(), X_y) 
    p = -1*np.ones((self.n))
    A = np.concatenate((np.eye(self.n), -1*np.eye(self.n)), axis=0)
    ones_n = np.ones((self.n))
    b = np.concatenate(((1/(tau*self.n))* ones_n, np.zeros((self.n))), axis=0)
    return Q, p, A, b

  def svm_solver(self, solver="mine"):
    if solver == "mine":
      x_sol, _, _ = self.barr_method()
    elif solver == "cvxopt":
      x_sol = self.cvxopt_solver()
    if not self.dual:
      # the hyperplane is explicit on the variables
      w = x_sol[:self.d]
    elif self.dual:
      # we can recover the hyperplane with a simple linear combination
      # w = \sum_{i=1}^m\lambda_iy_ix_i
      w = (np.sum(self.X * np.expand_dims(x_sol, 0) *
           np.expand_dims(self.y, 0), axis=1))
    acc = self.compute_accuracy(self.X, self.y, w)
    return x_sol, w, acc

  def compute_accuracy(self, X, y, w):
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


###############################################################################
#  Kernel-SVM
###############################################################################


class kernel_SVM(QP_solver):
  def __init__(self, tau, K, y, dual=True, squared=False):
    self.n = K.shape[0]
    self.y = y
    self.dual = dual
    self.K = K
    mu = 3.
    tol = 1e-1
    LS = True
    if dual:
      if squared:
        Q, p, A, b = self.transform_squared_svm_dual(tau, K, y)
      else:
        Q, p, A, b = self.transform_svm_dual(tau, K, y)
      x_0 = (1/(2*tau*self.n))*np.ones((self.n))
    elif not dual:
      Q, p, A, b = self.transform_svm_primal(tau, K, y)
      x_0 = 10 * np.ones((2*self.n))
      x_0[:self.n] = 0
    QP_solver.__init__(self, Q, p, A, b, x_0, mu, tol, t0=1.2, LS=True)

  ##############################################
  #  Hinge Loss
  ##############################################

  def transform_svm_primal(self, tau, K, y):
    K_y = K * np.expand_dims(y, 0)  # n x n
    A_1 = np.concatenate((K_y.T, np.eye(self.n)), axis=1)
    A_2 = np.concatenate((np.zeros((self.n, self.n)), np.eye(self.n)), axis=1)
    A = -1 * np.concatenate((A_1, A_2), axis=0)
    b = -1 * np.concatenate((np.ones((self.n)), np.zeros((self.n))), axis=0)
    Q_1 = np.concatenate((2 * K, np.zeros((self.n, self.n))), axis=1)
    Q_2 = np.zeros((self.n, self.n + self.n))
    Q = np.concatenate((Q_1, Q_2), axis=0)
    p = (np.concatenate((np.zeros((self.n)), (1/(tau*self.n)) *
         np.ones((self.n))), axis=0))
    pdb.set_trace()
    return Q, p, A, b

  def transform_svm_dual(self, tau, K, y):
    Q = np.dot(np.expand_dims(y,1), np.expand_dims(y,0)) * K
    p = -1*np.ones((self.n))
    A = np.concatenate((np.eye(self.n), -1*np.eye(self.n)), axis=0)
    ones_n = np.ones((self.n))
    b = np.concatenate(((1/(tau*self.n))* ones_n, np.zeros((self.n))), axis=0)
    return Q, p, A, b

  ##############################################
  #  Squared Hinge Loss
  ##############################################

  def transform_squared_svm_dual(self, tau, K, y):
    Q = np.dot(np.expand_dims(y,1), np.expand_dims(y,0))
    Q = Q * (K + self.n * tau * np.eye(self.n))
    p = -1*np.ones((self.n))
    A = -1*np.eye(self.n)
    b = np.zeros((self.n))
    return Q, p, A, b

  def svm_solver(self, solver="mine"):
    if solver == "mine":
      x_sol, _, _ = self.barr_method()
    elif solver == "cvxopt":
      x_sol = self.cvxopt_solver()
    if not self.dual:
      alpha = x_sol[:self.n]
    elif self.dual:
      lambd = x_sol
      alpha = lambd * self.y  # pointwise product
    acc = self.compute_accuracy(self.K, self.y, alpha)
    return x_sol, alpha, acc

  def compute_accuracy(self, K, y, alpha):
    # K is a (ntr x nte) matrix
    y_pred = np.dot(np.expand_dims(alpha, 0), K).T
    y_pred = y_pred[:, 0]
    # y_pred = self.Kernel.predict(self.X, X, alpha)
    correct = ((y * y_pred) >= 0)
    acc = np.mean(correct)
    return acc

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