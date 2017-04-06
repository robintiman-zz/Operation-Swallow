import load_files as lf
import vectorizer as vectorizer

traincsv = lf.load_csv('train.csv')
glove = lf.load_glove('glove.6B.50d.txt')

train_vec = vectorizer.vectorize(traincsv, glove)