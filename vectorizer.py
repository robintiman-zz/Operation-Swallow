import regex as re
import numpy as np
import spelling as sp
import time

"""
Vectorizing each question of the the csv_array using glove
Return: A vectorized csv_vec matrix where rows are each example and columns are the data the with format:
            [][0]: Correct value
            [][1]: Row number
            [][2]: Question 1 ID
            [][3]: Question 2 ID
            [][4-dim]: N-dimensional vector of vectorized question 1
            [][4+dim-end]: N-dimensional vector of vectorized question 2
"""
def vectorize(csv_array, glove, dim):
    csv_vec = np.zeros((len(csv_array), len(csv_array[0]) + 2*dim - 2))
    percentage = 0
    start = time.time()
    totaltime = 0

    for i in range(1, len(csv_array)):

        # Calculate % complete and estimated time left
        if i%int((len(csv_array)/100)) == 0:
            totaltime += (time.time()-start)
            time_estimate = totaltime/(percentage+1)*(100-percentage)
            min = int(time_estimate/60)
            sec = int(time_estimate - min*60)
            print("Vectorizing..." + str(percentage) + "% complete. Estimated time: " + str(min) + ":" + str(sec))
            start = time.time()
            percentage += 1

        # Lower case each question
        q1 = str.lower(csv_array[i][3])
        q2 = str.lower(csv_array[i][4])

        # Correct spelling
        #q1 = sp.correct_spelling(q1,q2)[0]
        #q2 = sp.correct_spelling(q1,q2)[1]

        # Regex seperating each word of the questions in a vector
        q1_words = re.findall(r'\p{L}+', q1)
        q2_words = re.findall(r'\p{L}+', q2)

        # Initialize numpy vectors
        q1_vec = np.zeros((1, dim))
        q2_vec = np.zeros((1, dim))

        # Add all vectorized words from glove in each question vector. If wrongly spelled, ignore it
        for word in q1_words:
            try:
                q1_vec = np.add(q1_vec, glove[word])
            except KeyError:
                continue
        for word in q2_words:
            try:
                q2_vec = np.add(q2_vec, glove[word])
            except KeyError:
                continue

        # Normalize the vectorized questions
        q1_vec = np.divide(q1_vec, np.linalg.norm(q1_vec))
        q2_vec = np.divide(q2_vec, np.linalg.norm(q2_vec))

        # Put together the whole final matrix
        csv_vec[i][0] = csv_array[i][5]
        csv_vec[i][1] = csv_array[i][0]
        csv_vec[i][2] = csv_array[i][1]
        csv_vec[i][3] = csv_array[i][2]
        for j in range(0, dim):
            csv_vec[i][j+4] = q1_vec[0][j]
        for j in range(0, dim):
            csv_vec[i][j+4+dim] = q2_vec[0][j]

    return csv_vec