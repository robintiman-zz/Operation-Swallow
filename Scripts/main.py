import load_files as lf
import vectorize as vec
import numpy as np
from libsvm import convert_to_libsvm
import pandas as pd
import LSTM_vectorizer as LSTM_vec

# Uncomment this if you're running it for the first time
dim = 50

# GloVe
glove = lf.load_glove("../Data/glove.6B.50d.txt")

# Training set
traindata = pd.read_csv('../Data/train.csv')
traindata = traindata.replace(np.nan, '', regex=True)
#vec.vectorize(dim, glove, traindata, is_train=True)        #Ordinary vectorizing
LSTM_vec.vectorize(dim, glove, traindata, is_train=True)   #LSTM vectorizing, takes up a lot of ram and disk!

# Test set
#testdata = pd.read_csv('../Data/test.csv')
#testdata = testdata.replace(np.nan, '', regex=True)
#vec.vectorize(dim, glove, testdata, is_train=False)        #Ordinary vectorizing
#LSTM_vec.vectorize(dim, glove, testdata, is_train=False)    #LSTM vectorizing, takes up a lot of ram and disk!

#
# # Load np files if already vectorized
# # train_vector = np.load("../Data/train_vector.npy")
# # test_vector = np.load("../Data/test_vector.npy")
# convert_to_libsvm(train_vector, traindata, True)
# convert_to_libsvm(test_vector, testdata, False)

# lf.conv_to_csv("../Data/pred.txt")

