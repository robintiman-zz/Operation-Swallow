import regex as re
import numpy as np

#Vectorizing each row of the the csv_file using glove
#Return: A csv_vec matrix with [0] as array of question 1 vectors, [1] as array of question 2 vectors
#           and [2] as array of correct match {0,1}
def vectorize(csv_file, glove):
    csv_vec = [[]]
    csv_vec.append([])
    csv_vec.append([])
    for row in csv_file:
        q1 = str.lower(row[3])
        q2 = str.lower(row[4])
        q1_words = re.findall(r'\p{L}+', q1)
        q2_words = re.findall(r'\p{L}+', q2)

        q1_vec = np.zeros((1, len(glove['hello'])))
        q2_vec = np.zeros((1, len(glove['hello'])))

        for word in q1_words:
            q1_vec = np.add(q1_vec, glove[word])
        for word in q2_words:
            q2_vec = np.add(q2_vec, glove[word])

        q1_vec = np.divide(q1_vec, np.linalg.norm(q1_vec))
        q2_vec = np.divide(q2_vec, np.linalg.norm(q2_vec))

        csv_vec[0].append(q1_vec)
        csv_vec[1].append(q2_vec)
        csv_vec[2].append(row[5])
    return csv_vec