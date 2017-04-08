import csv
import sys
import numpy as np

# Load glove file
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

# Load csv file
def load_csv(csvfile1):
    print("Loading csv file")
    ifile = open(csvfile1, "r")
    reader = csv.reader(ifile)
    data = list(reader)
    print("Done loading csv file into data list")
    ifile.close()
    return data