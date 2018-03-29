import os
import pdb
import numpy as np
import pandas as pd

submission_n = 3
path_save = ("/home/alexnowak/DataChallenge-KernelMethods/"
             "Data/submission_{}.csv".format(submission_n))
Datasets = [0, 1, 2]
Y = np.zeros([3000])
for i, dataset in enumerate(Datasets):
  path_submission = ("/home/alexnowak/DataChallenge-KernelMethods/"
                     "Data/dataset_{}/submission.npz".format(dataset))
  pred = np.load(path_submission)["Y"]
  info = np.load(path_submission)["info"]
  print("\nFor dataset {} :\n".format(dataset))
  print(info)
  Y[i*1000:(i+1)*1000] = pred.astype(int)
Submission = pd.DataFrame(Y, columns=['Bound']).astype('int')
# Submission['Bound'] = Submission['Bound'].astype('int')
path_read = ("/home/alexnowak/DataChallenge-KernelMethods/"
             "Data/dataset_0/Ytr0.csv")
ex = pd.DataFrame.from_csv(path_read)
Submission.to_csv(path_save, header = ["Bound"])