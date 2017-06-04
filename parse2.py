from __future__ import division
import numpy as np
import pandas as pd
from collections import Counter
import inflect

def get_weight(count, eps=5000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

def q_to_words(q):
	sentence =  ''.join([x.lower().replace(","," ") for x in q if x.isalnum() or x==" "]).split()
	s = []
	for w in sentence:
		if w.isdigit():
			s.extend(to_written_number(w))
		else:
			s.append(w)
	return s
	


p = inflect.engine()

def to_written_number(w):
	if w.isdigit():
		return p.number_to_words(w)
	else:
		return w

class Weighter(object):
	def __init__(self, word_embeddings, first_n_questions = 404246):
		print "Setting up weighter"
		self.we = word_embeddings
		
		df_train = pd.read_csv('data/train.csv')
		train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
		
		words = []
		[words.extend(q_to_words(q)) for q in train_qs]

		print "Counting words"
		counts = Counter(words)
		self.weights = {word: get_weight(count) for word, count in counts.items()}
		print "Counted words, weighter set up"
		
		self.arr = np.zeros((first_n_questions, 1200))
		self.is_duplicate = np.zeros(first_n_questions)
		self.i=0
		
	def weighted(self,q):
		x = [self.we.embeddings[self.we.words[w]]*self.weights[w] for w in q if self.we.has_word(w)]
		if len(x)!=0:
			return sum(x)/len(x)
		print "emtpy!"
		return np.zeros(300)

	def unweighted(self,q):
		x = [self.we.embeddings[self.we.words[w]] for w in q if self.we.has_word(w)]
		if len(x)!=0:
			return sum(x)/len(x)
		print "emtpy!"
		return np.zeros(300)

	def process(self, q1, q2, is_duplicate):
		self.arr[self.i]=np.concatenate(( self.weighted(q1),self.weighted(q2), self.unweighted(q1), self.unweighted(q2)))
		self.is_duplicate[self.i]=is_duplicate
		self.i+=1

