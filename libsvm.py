"""
Method for converting an array to the LIBSVM format
param: data:
    Will be on the format
    [label, q1_id, q2_id, q1f1, ... , q1fd, q2f1, ... , q2fd]
    where d is the number of dimensions
    
    is_train:
    True is the data is training set, False otherwise

output: libsvm_data
    A text file formatted according to the LIBSVM format
    [label, 1:value, 2:value, ... d:value]
    where d is the number of dimensions
"""
def convert_to_libsvm(data, is_train):
    print("Converting data to LIBSVM format")
    if is_train:
        file = open("datalib.txt.train", 'w')
        label_index = 0
        rowcount_index = 1
    else:
        file = open("datalib.txt.test", 'w')
        rowcount_index = 0
    nbr_samples = len(data)
    for row in data:
        rowcount = int(row[rowcount_index])
        print("Progress {:2.1%}".format(rowcount/nbr_samples), end="\r") # Works in the terminal. Sadly not in PyCharm
        if is_train:
            label = str(int(row[label_index]))
            write_file(row, file, is_train, label)
        else:
            write_file(row, file, is_train)
    file.close()
    print("LIBSVM formatted file created as 'datalib.txt.train'")


def write_file(row, file, is_train, label=""):
    if is_train:
        file.write(label + " ")
        data_index = 4
    else:
        data_index = 1
    for i in range(data_index, len(row)):
        value = row[i]
        file.write(str(i - data_index + 1) + ":" + str(value) + " ")
    file.write("\n")


