import load_files as lf
import vectorizer as vectorizer

dim = 50
traindata = lf.load_csv('train.csv')
glove = lf.load_glove('glove.6B.' + str(dim) + 'd.txt')

train_vec = vectorizer.vectorize(traindata, glove, dim)