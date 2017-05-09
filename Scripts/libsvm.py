import pandas as pd
import numpy as np

"""
Method for converting an array to the LIBSVM format
param: data:
    Will be on the format
    [label, q1_id, q2_id, q1f1, ... , q1fd, q2f1, ... , q2fd]
    where d is the number of dimensions
    
    is_train:
    True if the data is training set, False otherwise

output: libsvm_data
    A text file formatted according to the LIBSVM format
    [label, 1:value, 2:value, ... d:value]
    where d is the number of dimensions
"""
def convert_to_libsvm(data, csv, is_train):
    print("Converting data to LIBSVM format")
    if is_train:
        file = open("../Data/datalib.txt.train", 'w')
        labels = csv.is_duplicate.values
        id = csv.id.values
    else:
        file = open("../Data/datalib.txt.test", 'w')
        id = csv.test_id.values

    nbr_samples = len(data)
    for i in range(0, data.shape[0]):
        row = data[i, :]
        print("Progress {:2.1%}".format(i/nbr_samples), end='\r') # Works in the terminal. Sadly not in PyCharm
        if is_train:
            label = str(labels[i])
            write_file(row, file, is_train, label)
        else:
            write_file(row, file, is_train)
    file.close()
    print("LIBSVM formatted file created as 'datalib.txt.train'")

"""
Splits the training set into two. Useful for tuning parameters
"""
def split_to_libsvm(data):
    train = open("datalib.txt.train", 'w')
    test = open("datalib.txt.test", 'w')
    nbr_samples = len(data)
    for row in data:
        rowcount = int(row[1])
        print("Progress {:2.1%}".format(rowcount / nbr_samples), end="\r")  # Works in the terminal. Sadly not in PyCharm
        label = str(int(row[0]))
        if rowcount >= int(0.6*nbr_samples):
            write_file(row, train, True, label)
        else:
            write_file(row, test, True, label)
    test.close()
    train.close()

def write_file(row, file, is_train, label=""):
    if is_train:
        file.write(label + " ")
    for i in range(0, len(row)):
        value = row[i]
        file.write(str(i + 1) + ":" + str(value) + " ")
    file.write("\n")


