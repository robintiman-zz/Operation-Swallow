import enchant
from enchant.checker import SpellChecker

def __check_spelling(q1, q2):
    dict = enchant.Dict('en_US')
    sent1 = q1.split()
    sent2 = q2.split()
    for word in sent1:
        # Check each error
        if not dict.check(word):
            # If the word exists in the other sentence, it's likely to be correct
            if word in sent2:
                continue

            # If not in the other sentence. Check if any of the suggestions is.
            suggestions = dict.suggest(word)
            for s in suggestions:
                if s in sent2:
                    q1 = q1.replace(word, s)
    return q1

def correct_spelling(q1, q2):
    q1 = __check_spelling(q1, q2)
    q2 = __check_spelling(q2, q1)
    return q1, q2

