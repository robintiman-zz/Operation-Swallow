def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def loadTrainingData(trainingFile):

    return trainingData

trainingData = loadTrainingData('training.txt')
model = loadGloveModel('glove.txt')
print(model['hello'])