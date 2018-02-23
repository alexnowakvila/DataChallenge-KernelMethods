import os
import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pdb
from utils import *
import matplotlib.pyplot as plt

def phi(x, t, Q, p, A, b):
  phi_1 = t*(0.5*np.dot(np.dot(x.transpose(), Q), x) + np.dot(p.transpose(), x))
  dif = b - np.dot(A, x)
  barrier = -1 * np.sum(np.log(dif))
  phi = phi_1 + barrier
  return phi

def grad(x, t, Q, p, A, b):
  grad_phi_1 = t*(np.dot(Q,x) + p)
  dif = b - np.dot(A, x)
  grad_barrier = np.sum(A / np.expand_dims(dif, 1), axis=0)
  return grad_phi_1 + grad_barrier

def hess(x, t, Q, p, A, b):
  hess_phi_1 = t * Q
  dif = b - np.dot(A, x)
  A_1 = A / np.expand_dims(dif, 1)
  hess_barrier = np.dot(A_1.transpose(), A_1)
  return hess_phi_1 + hess_barrier

def dampedNewtonStep(x, f, g, h):
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

def dampedNewton(x0, f, g, h, tol, LS=False):
  max_tol = (3 - np.sqrt(5))/2
  try:
    assert tol < max_tol
  except:
    raise ValueError('tolerance must be smaller than {}'.format(max_tol))
  # compute first step
  xnew, gap = dampedNewtonStep(x0, f, g, h)
  xhist = [xnew]
  while gap > tol:
    if LS:
      xnew, gap = LSNewtonStep(xnew, f, g, h)
    else:
      xnew, gap = dampedNewtonStep(xnew, f, g, h)
    xhist.append(xnew)
  xstar = xnew
  return xstar, xhist

def linesearch(x, deltax, gx, f, g, alpha=0.4, beta=0.9):
  gx = g(x)
  d = np.dot(gx.transpose(), deltax)
  # T = np.arange(-2, 2, 0.001)
  # F1, F2, F3 = [],[], []
  # for t in list(T):
  #   F1.append(f(x + t*deltax))
  #   F2.append( (f(x) + alpha*t*d) )
  #   F3.append( (f(x) + t*d) )
  # plt.plot(list(T), F1, 'r')
  # plt.plot(list(T), F2, 'b')
  # plt.plot(list(T), F3, 'k')
  # plt.show()
  t = 1
  i = 0
  while np.isnan(f(x + t*deltax)) or f(x + t*deltax) > (f(x) + alpha*t*d):
    t *= beta
    i += 1
  return t

def LSNewtonStep(x, f, g, h):
  fx, gx, hx = f(x), g(x), h(x)
  hx_inv = np.linalg.inv(hx)
  lambd = np.sqrt(np.dot(np.dot(gx.transpose(), hx_inv), gx))
  deltax = -1 * np.dot(hx_inv, gx)
  lr = linesearch(x, deltax, gx, f, g)
  xnew = x + lr*deltax
  gap = 0.5*lambd*lambd
  return xnew, gap

def transform_svm_primal(tau, X, y):
  n = y.shape[0]
  d = X.shape[0]
  X_y = X * np.expand_dims(y, 0)  # d x n
  A_1 = np.concatenate((X_y.transpose(), np.eye(n)), axis=1)
  A_2 = np.concatenate((np.zeros((n, d)), np.eye(n)), axis=1)
  A = -1 * np.concatenate((A_1, A_2), axis=0)
  b = -1 * np.concatenate((np.ones((n)), np.zeros((n))), axis=0)
  Q_1 = np.concatenate((np.eye(d), np.zeros((d, n))), axis=1)
  Q_2 = np.zeros((n, d + n))
  Q = np.concatenate((Q_1, Q_2), axis=0)
  p = np.concatenate((np.zeros((d)), (1/(tau*n)) * np.ones((n))), axis=0)
  return Q, p, A, b

def transform_svm_dual(tau, X, y):
  n = y.shape[0]
  X_y = X * np.expand_dims(y, 0)  # d x n
  Q = np.dot(X_y.transpose(), X_y) 
  p = -1*np.ones((n))
  A = np.concatenate((np.eye(n), -1*np.eye(n)), axis=0)
  ones_n = np.ones((n))
  b = np.concatenate(((1/(tau*n))* ones_n, np.zeros((n))), axis=0)
  return Q, p, A, b

def barr_method(Q, p, A, b, x_0, mu, tol, t0=1.2, LS=False):
  t = t0
  m = A.shape[0]
  x0 = x_0
  xhist = [x0]
  barriers = []
  while m/t > tol:
    f = lambda x: phi(x, t, Q, p, A, b)
    g = lambda x: grad(x, t, Q, p, A, b)
    h = lambda x: hess(x, t, Q, p, A, b)
    xstar, xhist_dn = dampedNewton(x0, f, g, h, tol, LS=LS)
    xhist.extend(xhist_dn)
    barriers.extend([x0 for i in xhist_dn])
    x0 = xstar
    t *= mu
  x_sol = xstar
  return x_sol, xhist, barriers

def svm_solver(tau, X, Y, t0, mu, tol, LS= False,
               model='primal', solver="mine"):
  n = X.shape[1]
  d = X.shape[0]
  if model == 'primal':
    Q, p, A, b = transform_svm_primal(tau, X, Y)
    x_0 = 10 * np.ones((d + n))
    x_0[:d] = 0
  elif model == 'dual':
    Q, p, A, b = transform_svm_dual(tau, X, Y)
    x_0 = (1/(2*tau*n))*np.ones((n))
  else:
    raise ValueError('Must correctly specify the model (primal/dual)')
  if solver == "mine":
    x_sol, xhist, barriers = barr_method(Q, p, A, b, x_0,
                                         mu, tol, t0=t0, LS=LS)
    f = (lambda x: 0.5*np.dot(np.dot(x.transpose(), Q), x) + 
       np.dot(p.transpose(), x))
    fhist = np.array([f(x) for x in xhist])
    fhist_b = np.array([f(x) for x in barriers])
  elif solver == "cvxopt":
    Q, p, A, b = matrix(Q), matrix(p), matrix(A), matrix(b)
    solution = cvxopt.solvers.qp(Q, p, A, b)
    x_sol = np.array(solution['x'])[:,0]
    xhist = []
    barriers = []
    fhist = []
    fhist_b = []
  if model == 'primal':
    # the hyperplane is explicit on the variables
    w = x_sol[:d]
  elif model == 'dual':
    # we can recover the hyperplane with a simple linear combination
    # w = \sum_{i=1}^m\lambda_iy_ix_i
    w = np.sum(X * np.expand_dims(x_sol, 0) * np.expand_dims(Y, 0), axis=1)
  acc = compute_accuracy(X, Y, w)
  return x_sol, xhist, fhist, fhist_b, w, acc


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