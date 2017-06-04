"""
from nltk.corpus import wordnet as wn

def get_synonym(word):
    syns = []
    try:
        [syns.extend(x.lemmas()) for x in wn.synsets(word)]
        w = random.choice(syns).name()
        if not w.isalnum():
            return word
        return w
    except Exception as e:
        print e
        return word

def get_all_synonyms(word):
    syns = []
    [syns.extend(x.lemmas()) for x in wn.synsets(word)]
    return [x.name() for x in syns]
"""
