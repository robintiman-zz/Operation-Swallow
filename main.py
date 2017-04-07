import load_files as lf
import vectorizer as vec
import libsvm

dim = 50
traindata = lf.load_csv('train.csv')
glove = lf.load_glove('glove.6B.' + str(dim) + 'd.txt')

train_vec = vec.vectorize(traindata, glove, dim)
libsvm.convert_to_libsvm(train_vec)