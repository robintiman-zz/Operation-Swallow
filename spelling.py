from enchant.checker import SpellChecker
import enchant

def check_spelling(q1, q2):
    dict = enchant.Dict('en_US')
    sent1 = q1.split()
    sent2 = q2.split()
    # Check each word
    for word in sent1:
        if not dict.check(word):
            # If the word exists in the other sentence, it's likely to be correct
            if word in sent2:
                break

            # If not in the other sentence. Check if any of the suggestions is.
            suggestions = dict.suggest(word)

            for s in suggestions:
