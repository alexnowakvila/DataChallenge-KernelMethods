# Data Challenge - Kernel Methods (MVA)
Authors: Adrien Le Franc and Alex Nowak

## Introduction
The goal of the data challenge is to learn how to implement machine learning algorithms, gain understanding about them and adapt them to structural data. For this reason, we have chosen a sequence classification task: predicting whether a DNA sequence region is binding site to a specific transcription factor.

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes. Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound. In this challenge, we will work with three datasets corresponding to three different TFs.

## What is expected
Two days after the deadline of the data challenge, you will have to provide

- a small report on what you did (in pdf format, 11pt, 2 pages A4 max)
- your source code (zip archive), with a simple script "start" (that may be called from Matlab, Python, R, or Julia) which will reproduce your submission and saves it in Yte.csv

## Rules
- At most 3 persons per team.
- One team can submit results up to twice per day during the challenge.
- A leader board will be available during the challenge, which shows the best results per team, as measured on a subset of the test set. A different part of the test set will be used after the challenge to evaluate the results.
- Registration has to be done with email addresses @ens-cachan.fr, @polytechnique.edu, @u-psud.fr, @student.ecp.fr, @ens.fr, @mines-paristech.fr, @telecom-paristech.fr, @ensiee.fr, @dauphine.eu, @centralesupelec.fr, @ensiie.fr, @etu.parisdescartes.fr, @ens-paris-saclay.fr, @eleves.enpc.fr, @mines-ensae.fr.
- The most important rule is: DO IT YOURSELF. The goal of the data challenge is not get the best score on this data set at all costs, but instead to learn how to implement things in practice, and gain practical experience with the machine learning techniques involved.


**For this reason, the use of external machine learning libraries is forbidden. For instance, this includes, but is not limited to, libsvm, liblinear, scikit-learn, ...**

On the other hand, you are welcome to use general purpose libraries, such as library for linear algebra (e.g., svd, eigenvalue decompositions), optimization libraries (e.g., for solving linear or quadratic programs)
