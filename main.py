import csv
import sys

def loadGlove(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
            embedding = [float(val) for val in splitLine[1:]]
        except(ValueError):
            print(line)
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    f.close()
    return model

def readCSV(csvfile1):
    print("Loading csv file")
    csv.field_size_limit(sys.maxsize)
    csvfile2 = open(csvfile1, 'r')
    csvfile3 = csv.reader(csvfile2, delimiter=',', quotechar='|')
    print("Done loading csv")
    return csvfile3

traincsv = readCSV('train.csv')
glove = loadGlove('glove.840B.300d.txt')
print(glove['hello'])