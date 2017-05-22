import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter as mset
from nltk.corpus import stopwords
import scipy as sp

# Load data from csv files
traindata = pd.read_csv('../Data/train.csv')
traindata = traindata.replace(np.nan, '', regex=True)
labels = traindata.is_duplicate.values
nbr_duplicates = np.sum(labels)
nbr_nonduplicates = len(labels) - nbr_duplicates
# testdata = pd.read_csv('../Data/test.csv')
q1_arr = traindata.question1.values
q2_arr = traindata.question2.values


def duplicate_ratio():
    y_train = traindata.is_duplicate.values
    plt.hist(y_train, bins=3)
    plt.title("Histogram showing the is_duplicates column")
    plt.show()


def word_length(with_spaces):
    length = len(q1_arr)
    same_label = np.zeros((length, 1))
    different_label = np.zeros((length, 1))
    same_index = 0
    different_index = 0
    for i in range(0, length):
        q1 = q1_arr[i]
        q2 = q2_arr[i]
        # To handle empty strings. q2 with id 105780 is one
        if q1 == q1:
            if not with_spaces:
                q1  = q1.replace(" ", "")
            q1_size = len(q1)
        else:
            q1_size = 0
        if q2 == q2:
            if not with_spaces:
                q2 = q2.replace(" ", "")
            q2_size = len(q2)
        else:
            q2_size = 0
        if labels[i] == 1:
            # Same label
            same_label[same_index, 0] = abs(q1_size - q2_size)
        else:
            different_label[different_index, 0] = abs(q1_size - q2_size)
    bins = np.linspace(0, 140, 20)
    s = "with" if with_spaces else "without"
    plt.title("Question length " + s + " spaces")
    plt.hist(same_label, bins, alpha=0.5, label="is_duplicate=1")
    plt.hist(different_label, bins, alpha=0.5, label="is_duplicate=0")
    plt.legend(loc='upper right')
    plt.savefig('question_length_' + s + '_spaces')
    plt.show()


def common_words(title, save_title, with_spelling, remove_stopwords):
    same = np.zeros((nbr_duplicates, 1))
    different = np.zeros((nbr_nonduplicates, 1))
    same_index = 0
    different_index = 0
    all_is_stop_count = 0
    for i in range(0, len(q1_arr)):
        str1 = q1_arr[i]
        str2 = q2_arr[i]
        q1 = str_to_array(str1)
        q2 = str_to_array(str2)
        if remove_stopwords:
            q1 = remove_stop(q1)
            q2 = remove_stop(q2)

        # Finds the common words
        common = list((mset(q1) & mset(q2)).elements())
        if len(q1) + len(q2) == 0:
            all_is_stop_count += 1
            continue

        if labels[i] == 0:
            different[different_index, 0] = len(common)/max(len(q1), len(q2))
            different_index += 1
        else:
            same[same_index, 0] = len(common)/max(len(q1), len(q2))
            same_index += 1
    bins = np.linspace(0, 1, 100)
    print(all_is_stop_count)
    plt.title(title)
    plt.hist(same, bins, alpha=0.5, label="is_duplicate=1")
    plt.hist(different, bins, alpha=0.5, label="is_duplicate=0")
    plt.legend(loc='upper right')
    plt.savefig('../Visualization/' + save_title)
    plt.show()


def str_to_array(str):
    return re.findall("\w+", str.lower())


def remove_stop(q):
    stop = set(stopwords.words('english'))
    result = [word for word in q if word not in stop]
    return result

"""
Removes common words and returns the resulting arryas along with the number of common words. 
"""
def remove_common(q1, q2):
    common = list((mset(q1) & mset(q2)).elements())
    q1 = [word for word in q1 if word not in common]
    q2 = [word for word in q2 if word not in common]
    return q1, q2, len(common)

def vector_distance(title, save_title, glove, metric='euclidean'):
    same = np.zeros((nbr_duplicates, 1))
    diff = np.zeros((nbr_nonduplicates, 1))
    same_index = 0
    diff_index = 0
    not_in_glove = 0
    in_glove = 0
    words_not_found = []
    for i in range(0, len(q1_arr)):
        if i % 1000 == 0:
            print('{0:.2f}% finished'.format(i/len(q1_arr)*100))

        q1 = str_to_array(q1_arr[i])
        q2 = str_to_array(q2_arr[i])

        q1 = remove_stop(q1)
        q2 = remove_stop(q2)

        q1_vec = np.zeros((1, 50))
        q2_vec = np.zeros((1, 50))
        # Add all vectorized words from glove in each question vector. If wrongly spelled, ignore it
        for word in q1:
            try:
                q1_vec = np.add(q1_vec, glove[word])
                in_glove += 1
            except KeyError:
                words_not_found.append(word)
                # print("{0} in {1}\nOther sentence: {2}".format(word, q1_arr[i], q2_arr[i]))
                not_in_glove += 1
                continue
        for word in q2:
            try:
                q2_vec = np.add(q2_vec, glove[word])
                in_glove += 1
            except KeyError:
                # print("{0} in {1}\nOther sentence: {2}".format(word, q2_arr[i], q1_arr[i]))
                words_not_found.append(word)
                not_in_glove += 1
                continue

        # Normalize the vectorized questions
        q1_norm = np.linalg.norm(np.transpose(q1_vec), np.inf)
        q2_norm = np.linalg.norm(np.transpose(q2_vec), np.inf)
        if q1_norm > 0 and q2_norm > 0:
            q1_vec = np.divide(q1_vec, q1_norm)
            q2_vec = np.divide(q2_vec, q2_norm)

        dist = sp.spatial.distance.cdist(q1_vec, q2_vec, metric)
        if np.isnan(dist):
            continue
        if labels[i] == 1:
            same[same_index, 0] = dist
            same_index += 1
        else:
            diff[diff_index, 0] = dist
            diff_index += 1
    np.save("../Data/vec_dist_same", same)
    np.save("../Data/vec_dist_diff", diff)
    bins = np.linspace(0, 4, 100)
    plt.title(title)
    plt.hist(same, bins, alpha=0.5, label="is_duplicate=1")
    plt.hist(diff, bins, alpha=0.5, label="is_duplicate=0")
    plt.legend(loc='upper right')
    plt.savefig('../Visualization/' + save_title)
    plt.show()

def not_found_in_glove(glove, title, save_title):
    # words = np.load("../Data/not_found.npy")
    same = np.zeros((nbr_duplicates, 2))
    diff = np.zeros((nbr_nonduplicates, 2))
    not_in_glove1 = 0
    not_in_glove2 = 0
    same_index = 0
    diff_index = 0
    for i in range(0, len(q1_arr)):
        if i % 1000 == 0:
            print('{0:.2f}% finished'.format(i/len(q1_arr)*100))

        q1 = str_to_array(q1_arr[i])
        q2 = str_to_array(q2_arr[i])

        q1 = remove_stop(q1)
        q2 = remove_stop(q2)

        for word in q1:
            try:
                glove[word]
            except KeyError:
                # print("{0} in {1}\nOther sentence: {2}".format(word, q1_arr[i], q2_arr[i]))
                not_in_glove1 += 1
                continue
        for word in q2:
            try:
                glove[word]
            except KeyError:
                # print("{0} in {1}\nOther sentence: {2}".format(word, q2_arr[i], q1_arr[i]))
                not_in_glove2 += 1
                continue

        q1_ratio = not_in_glove1/len(q1) if len(q1) > 0 else 0
        q2_ratio = not_in_glove2/len(q2) if len(q2) > 0 else 0
        if labels[i] == 1:
            # Duplicate
            same[same_index, 0] = q1_ratio
            same[same_index, 1] = q2_ratio
            same_index += 1
        else:
            diff[diff_index, 0] = q1_ratio
            diff[diff_index, 1] = q2_ratio
            diff_index += 1

        not_in_glove1 = 0
        not_in_glove2 = 0

    np.save("same_" + save_title, same)
    np.save("diff_" + save_title, diff)
    bins = np.linspace(0, 1, 100)
    plt.title(title)
    plt.hist(same[:, 0], bins, alpha=0.5, label="Q1. Duplicate")
    plt.hist(same[:, 1], bins, alpha=0.5, label="Q2. Duplicate")
    plt.hist(diff[:, 0], bins, alpha=0.5, label="Q1. Not duplicate")
    plt.hist(diff[:, 1], bins, alpha=0.5, label="Q2. Not duplicate")
    plt.legend(loc='upper right')
    plt.savefig('../Visualization/' + save_title)
    plt.show()

# glove = load_glove("../Data/glove.6B.50d.txt")
# # np.save("../Data/glove50d", glove)
glove = np.load("../Data/glove50d.npy").item()
# vector_distance("Vector distance", "vec_dist", glove, 'euclidean')
not_found_in_glove(glove, "Not found in GloVe", "not_found")


