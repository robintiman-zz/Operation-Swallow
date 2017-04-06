import csv
import sys

def load_glove(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = embedding
        except ValueError:
            print(line)
    print("Done.", len(model), " words loaded!")
    f.close()
    return model

def read_csv(csvfile1):
    print("Loading csv file")
    csv.field_size_limit(sys.maxsize)
    csvfile2 = open(csvfile1, 'r')
    csvfile3 = csv.reader(csvfile2, delimiter=',', quotechar='|')
    print("Done loading csv")
    return csvfile3