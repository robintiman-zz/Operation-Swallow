import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter as mset
import spelling as sp

# Load data from csv files
traindata = pd.read_csv('../Data/train.csv')
traindata = traindata.replace(np.nan, '', regex=True)
is_duplicates = traindata.is_duplicate.values
nbr_duplicates = np.sum(is_duplicates)
nbr_nonduplicates = len(is_duplicates) - nbr_duplicates
# testdata = pd.read_csv('../Data/test.csv')
q1_arr = traindata.question1.values
q2_arr = traindata.question2.values
labels = traindata.is_duplicate.values

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


def common_words(with_spelling):
    same = np.zeros((nbr_duplicates, 1))
    different = np.zeros((nbr_nonduplicates, 1))
    same_index = 0
    different_index = 0
    for i in range(0, len(q1_arr)):
        str1 = q1_arr[i].lower()
        str2 = q2_arr[i].lower()
        q1 = re.findall("\w+'?\w*", str1)
        q2 = re.findall("\w+'?\w*", str2)
        # Finds the common words
        words = list((mset(q1) & mset(q2)).elements())
        if labels[i] == 0:
            different[different_index, 0] = len(words)/max(len(q1), len(q2))
            different_index += 1
        else:
            same[same_index, 0] = len(words)/max(len(q1), len(q2))
            same_index += 1
    bins = np.linspace(0, 1, 100)
    plt.title("Common words")
    plt.hist(same, bins, alpha=0.5, label="is_duplicate=1")
    plt.hist(different, bins, alpha=0.5, label="is_duplicate=0")
    plt.legend(loc='upper right')
    plt.savefig('../Visualization/common_words')
    plt.show()

common_words(False)