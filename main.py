import load_files as lf
import vectorizer as vec
import libsvm
import numpy as np

# Uncomment this if you're running it for the first time
# dim = 50
# traindata = lf.load_csv('train.csv')
# glove = lf.load_glove('glove.6B.' + str(dim) + 'd.txt')
# train_vec = vec.vectorize(traindata, glove, dim)
# np.save("train_vec", train_vec)

train_vec = np.load("train_vec.npy")
libsvm.convert_to_libsvm(train_vec)