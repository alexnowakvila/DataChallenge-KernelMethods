import numpy as np
import time

import sys
sys.path.insert(0, '/home/adrien/MVA/KERNEL/DataChallenge-KernelMethods/Code/Alex/')

from utils import *
from svm import *


def K(s, t, gamma, N):
	"""
	s, t: input strings
	gamma: real value
	N: length of substrings
	
	compute substring kernel between s and t
	"""
	
	if N < 2:
		print('works only if N > 1')
		return False
	
	S = len(s)
	T = len(t)
	grid = np.ones((S, T, N))
	grid_K = np.zeros((S, T))
	
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


def normalize(K):
	
	n = K.shape[0]
	k = np.expand_dims(1./np.sqrt(np.diag(K)), 1)
	kkt = np.dot(k, k.T)
	K = K * kkt
	return K


def gram_matrix(X, substrings, l, k):
	"""
	X: input data matrix
	substrings: list of possible substrings
	l, k: parameter of kernel
	"""
	
	N = len(X)
	M = len(substrings)
	features = np.zeros((N, M))
	
	start = time.time()
	
	for i in tqdm(range(N)):
		for j in range(M):
			
			features[i, j] = K(X[i], substrings[j], l, k)
	
	K_matrix = np.zeros((N, N))
	
	for i in range(N):
	
		K_matrix[i, i] = features[i, :].T.dot(features[i, :]) 
		for j in range(i+1, N):

			K_matrix[i, j] = features[i, :].T.dot(features[j, :]) 
			K_matrix[j, i] = K_matrix[i, j]
	
	print('Gram matrix computed in {} min'.format((time.time() - start)/float(60)))
	
	return normalize(K_matrix)



if __name__ == "__main__":
	path_data = ('/home/adrien/MVA/KERNEL/DataChallenge-KernelMethods/Data/')
	Dataset0 = read_data(path_data, dataset=0)
	Dataset1 = read_data(path_data, dataset=1)
	Dataset2 = read_data(path_data, dataset=2)

	#############################################################################
	#  Create Substring Kernel
	#############################################################################

	dataset = 0
	k = 4
	gamma = 0.8
	path_save_kernel_mat = ("/home/alexnowak/DataChallenge-KernelMethods/"
						  "Data/dataset_{}/SubsKernel".format(dataset))

	alphabet = ['A', 'C', 'T', 'G']
	substrings = []

	for i1 in alphabet:
		for i2 in alphabet:
			for i3 in alphabet:
				for i4 in alphabet:
				
					if i1+i2+i3+i4 not in substrings:
						substrings.append(i1+i2+i3+i4)


	Xtr = Dataset0["Xtr_mat50"][:2000]
	Xte = Dataset0["Xtr_mat50"][:2000]

	Ktr = gram_matrix(Xtr, substrings, gamma, k)
	Kte = gram_matrix(Xte, substrings, gamma, k)
	np.savez(path_save_kernel_mat, Ktr=Ktr, Kte=Kte)
	print("Saved!")
