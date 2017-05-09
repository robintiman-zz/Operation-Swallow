import load_files as lf
import vectorize as vec
import numpy as np
from libsvm import convert_to_libsvm
import pandas as pd

# Uncomment this if you're running it for the first time
# dim = 50
#
# # GloVe
# glove = lf.load_glove("../Data/glove.6B.50d.txt")
#
# # Training set
# traindata = pd.read_csv('../Data/train.csv')
# traindata = traindata.replace(np.nan, '', regex=True)
# train_vector = vec.vectorize(dim, glove, traindata)
# np.save("train_vector", train_vector)
#
# # Test set
# testdata = pd.read_csv('../Data/test.csv')
# testdata = testdata.replace(np.nan, '', regex=True)
# test_vector = vec.vectorize(dim, glove, testdata)
# np.save("test_vector", test_vector)
#
# # Load np files if already vectorized
# # train_vector = np.load("train_vector.npy")
# # test_vector = np.load("test_vector.npy")
# convert_to_libsvm(train_vector, traindata, True)
# convert_to_libsvm(test_vector, testdata, False)

lf.conv_to_csv("../Data/pred.txt")

