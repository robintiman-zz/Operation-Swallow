"""
Method for converting an array to the LIBSVM format
param: data
    Will be on the format
    [label, q1_id, q2_id, q1f1, ... , q1fd, q2f1, ... , q2fd]
    where d is the number of dimensions

output: libsvm_data
    A text file formatted according to the LIBSVM format
    [label, 1:value, 2:value, ... d:value]
    where d is the number of dimensions
"""
def convert_to_libsvm(data):
    train = open("datalib.txt.train", 'w')
    test = open("datalib.txt.test", 'w')
    nbr_samples = len(data)
    print("Converting data to LIBSVM format")
    for row in data:
        label = str(int(row[0]))
        rowcount = int(row[1])
        if rowcount % 500 == 0:
            print(str(int(rowcount/nbr_samples*100)) + "%")
        if rowcount > int(0.7*nbr_samples):
            write_file(label, row, test)
        else:
            write_file(label, row, train)
    train.close()
    test.close()
    print("LIBSVM formatted file created as 'datalib.txt.train'")


def write_file(label, row, file):
    file.write(label + " ")
    for i in range(4, len(row)):
        value = row[i]
        file.write(str(i - 3) + ":" + str(value) + " ")
    file.write("\n")


