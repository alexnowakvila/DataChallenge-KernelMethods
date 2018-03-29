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