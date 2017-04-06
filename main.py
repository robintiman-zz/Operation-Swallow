import load_files as lf

traincsv = lf.read_csv('train.csv')
glove = lf.load_glove('glove.840B.300d.txt')
print(glove['hello'])