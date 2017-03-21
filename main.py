from numpy import genfromtxt

def parseFile(file):
    dataset = genfromtxt(file, delimiter=',')

    tempfile = open("voice.csv", 'r').readlines()
    for i in range(1, dataset.shape[0]):
        line = tempfile[i]
        if "female" in line:
            dataset[i][20] = 1
        elif "male" in line:
            dataset[i][20] = 0
    return dataset

dataset = parseFile('voice.csv')
print(dataset[1])