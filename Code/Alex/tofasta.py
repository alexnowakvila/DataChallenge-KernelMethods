import os
import pandas as pd
import numpy as np
import pdb
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna, generic_protein
from subprocess import Popen, PIPE, STDOUT, call

run = "training"
# run = "submission"
Datasets = [0, 1, 2]
# Datasets = [2]
for dataset in Datasets:
  if run == "training":
    data_path = ("/home/alexnowak/DataChallenge-KernelMethods/"
                 "Data/dataset_{}/".format(dataset))
    path = os.path.join(data_path, "Xtr{}.csv".format(dataset))
    with open(path, "r") as file:
      X = []
      for f in file:
        X.append(list(f[:-2]))
    Y = [''.join(x) for x in X]
    records = []
    for (index, seq) in enumerate(Y):
        records.append(SeqRecord(Seq(seq, generic_dna), str(index), description=""))
    path_shape = "/home/alexnowak/DataChallenge-KernelMethods/Code/Alex/shape/"
    path_save = os.path.join(path_shape, "Xtr{}.fa".format(dataset))
    SeqIO.write(records, path_save, "fasta")
    cmd = "Rscript {}{} {}".format(path_shape, "main.R", dataset)
    # p = Popen(cmd, shell=True)
    print("Running")
    call(cmd, shell=True)
    print("Done")

    #############################################################################
    #  Build Kernel matrix from shape features
    #############################################################################

    path_load = os.path.join(path_shape, "Xtr{}_shape.csv".format(dataset))
    Features = pd.read_csv(path_load).as_matrix()
    Ktr = np.dot(Features, Features.T)
    path_save_np = os.path.join(data_path, "ShapeKernel")
    np.savez(path_save_np, Ktr=Ktr)
  elif run == "submission":
    data_path = ("/home/alexnowak/DataChallenge-KernelMethods/"
                 "Data/dataset_{}/".format(dataset))
    X = []
    path = os.path.join(data_path, "Xtr{}.csv".format(dataset))
    with open(path, "r") as file:
      for f in file:
        X.append(list(f[:-2]))
    path = os.path.join(data_path, "Xte{}.csv".format(dataset))
    with open(path, "r") as file:
      for f in file:
        X.append(list(f[:-2]))
    Y = [''.join(x) for x in X]
    records = []
    for (index, seq) in enumerate(Y):
        records.append(SeqRecord(Seq(seq, generic_dna), str(index), description=""))
    path_shape = "/home/alexnowak/DataChallenge-KernelMethods/Code/Alex/shape/"
    path_save = os.path.join(path_shape, "Xtr{}.fa".format(dataset))
    SeqIO.write(records, path_save, "fasta")
    cmd = "Rscript {}{} {}".format(path_shape, "main.R", dataset)
    # p = Popen(cmd, shell=True)
    print("Running")
    call(cmd, shell=True)
    print("Done")

    #############################################################################
    #  Build Kernel matrix from shape features
    #############################################################################

    path_load = os.path.join(path_shape, "Xtr{}_shape.csv".format(dataset))
    Features = pd.read_csv(path_load).as_matrix()
    Ktr = np.dot(Features, Features.T)
    path_save_np = os.path.join(data_path, "ShapeKernel_all")
    np.savez(path_save_np, Ktr=Ktr)