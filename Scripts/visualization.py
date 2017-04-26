import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter as mset
from nltk.corpus import stopwords
import scipy as sp
from load_files import load_glove

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
    return re.findall("\w+'?\w*", str.lower())


def remove_stop(q):
    stop = set(stopwords.words('english'))
    result = [word for word in q if word not in stop]
    return result

def vector_distance(title, save_title, glove, metric='euclidean'):
    same = np.zeros((nbr_duplicates, 1))
    diff = np.zeros((nbr_nonduplicates, 1))
    same_index = 0
    diff_index = 0

    for i in range(0, len(q1_arr)):
        if i % 1000 == 0:
            print('{0:.2f}% finished'.format(i/len(q1_arr)*100))

        q1 = str_to_array(q1_arr[i])
        q2 = str_to_array(q2_arr[i])

        # Remove common words
        common = list((mset(q1) & mset(q2)).elements())
        q1 = [word for word in q1 if word not in common]
        q2 = [word for word in q2 if word not in common]

        q1 = remove_stop(q1)
        q2 = remove_stop(q2)

        q1_vec = np.zeros((1, 50))
        q2_vec = np.zeros((1, 50))
        # Add all vectorized words from glove in each question vector. If wrongly spelled, ignore it
        for word in q1:
            try:
                q1_vec = np.add(q1_vec, glove[word])
            except KeyError:
                continue
        for word in q2:
            try:
                q2_vec = np.add(q2_vec, glove[word])
            except KeyError:
                continue

        # Normalize the vectorized questions
        q1_norm = np.linalg.norm(q1_vec)
        q2_norm = np.linalg.norm(q2_vec)
        if q1_norm > 0 and q2_norm > 0:
            q1_vec = np.divide(q1_vec, q1_norm)
            q2_vec = np.divide(q2_vec, q2_norm)

        if labels[i] == 1:
            same[same_index, 0] = sp.spatial.distance.cdist(q1_vec, q2_vec, metric)
            same_index += 1
        else:
            diff[diff_index, 0] = sp.spatial.distance.cdist(q1_vec, q2_vec, metric)
            diff_index += 1

    np.save("vec_dist_same", same)
    np.save("vec_dist_diff", diff)
    bins = np.linspace(0, 20, 70)
    plt.title(title)
    plt.hist(same, bins, alpha=0.5, label="is_duplicate=1")
    plt.hist(diff, bins, alpha=0.5, label="is_duplicate=0")
    plt.legend(loc='upper right')
    plt.savefig('../Visualization/' + save_title)
    plt.show()

# common_words("Common words with stopwords", False, False)
# common_words("Common words without stopwords", "cmn_words_with_stop", False, True)

# glove = load_glove("../Data/glove.6B.50d.txt")
# np.save("../Data/glove50d", glove)
glove = np.load("../Data/glove50d.npy").item()
# vector_distance("Vector distance", "vec_dist", glove, 'cosine')
same = np.load("vec_dist_same.npy")
diff = np.load("vec_dist_diff.npy")
bins = np.linspace(0, 20, 70)
plt.title("Vec diff")
plt.hist(same, bins, alpha=0.5, label="is_duplicate=1")
plt.hist(diff, bins, alpha=0.5, label="is_duplicate=0")
plt.legend(loc='upper right')
plt.savefig('../Visualization/vec_dist')
plt.show()