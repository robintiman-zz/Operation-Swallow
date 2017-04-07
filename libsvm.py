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
    rowcount = 0
    for row in data:
        print(rowcount)
        count = 0
        if rowcount == 1000:
            break
        for value in row:
            if count == 0:
                label = str(value)
                f.write(label + " ")
            else:
                f.write(str(count) + ":" + str(value) + " ")
            count += 1
        count = 0
        rowcount += 1
        f.write("\n")
    f.close()


