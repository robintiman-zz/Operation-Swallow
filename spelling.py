import enchant
from enchant.checker import SpellChecker

def __check_spelling(q1, q2):
    dict = enchant.Dict('en_US')
    chkr = SpellChecker("en_US")
    chkr.set_text(q1)
    sent2 = q2.split()
    for err in chkr:
    # Check each error
        if not dict.check(err):
            # If the word exists in the other sentence, it's likely to be correct
            if err in sent2:
                continue

            # If not in the other sentence. Check if any of the suggestions is.
            suggestions = dict.suggest(err)
            for s in suggestions:
                if s in sent2:
                    q1 = q1.replace(err, s)
    return q1

def correct_spelling(q1, q2):
    q1 = __check_spelling(q1, q2)
    q2 = __check_spelling(q2, q1)
    return q1, q2

