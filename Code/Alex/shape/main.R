#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  dataset <- args[1]
}
dataset
library("DNAshapeR")

fn <- sprintf("/home/alexnowak/DataChallenge-KernelMethods/Code/Alex/shape/Xtr%s.fa", dataset)
# fn <- "sample.fa"
pred <- getShape(fn)

featureNames <- c()

###############################################################################
#  n-shape
###############################################################################
max_n <- 10

for (i in 1:max_n)
{
  featureNames <- append(featureNames, sprintf("%d-shape", i))
}

###############################################################################
#  n-MGW
###############################################################################

for (i in 1:max_n)
{
  featureNames <- append(featureNames, sprintf("%d-shape", i))
}

###############################################################################
#  n-ProT
###############################################################################

for (i in 1:max_n)
{
  featureNames <- append(featureNames, sprintf("%d-shape", i))
}

###############################################################################
#  n-Roll
###############################################################################

for (i in 1:max_n)
{
  featureNames <- append(featureNames, sprintf("%d-shape", i))
}

###############################################################################
#  n-HelT
###############################################################################

for (i in 1:max_n)
{
  featureNames <- append(featureNames, sprintf("%d-shape", i))
}

featureVector <- encodeSeqShape(fn, pred, featureNames, FALSE)
dim(featureVector)
path_save <- sprintf("/home/alexnowak/DataChallenge-KernelMethods/Code/Alex/shape/Xtr%s_shape.csv", dataset)
write.csv(featureVector, file = path_save)