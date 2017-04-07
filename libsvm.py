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
    f = open("datalib.txt", 'w')
    print("Converting data to LIBSVM format")
    for row in data:
        label = str(int(row[0]))
        rowcount = row[1]
        # if rowcount == 500:
        print(rowcount)
        f.write(label + " ")
        for i in range(4, len(row)):
            value = row[i]
            f.write(str(i - 3) + ":" + str(value) + " ")
        f.write("\n")
    f.close()


