import csv
import operator
import numpy as np
import cPickle
import network2
import random
from wn_helpers import *
from word_embeddings import *
from parse2 import *

class Dataset(object):
	def __init__(self,images,labels):
		self.images=images
		self.labels=labels

class SentenceToWords(object):
	def __init__(self,n,we):
		self.q1=[]
		self.q2=[]
		self.data_label=np.zeros(n)
		self.we = we
		self.i=0

	def process(self, q1, q2, is_duplicate):
		if not (len(q1)==0 or len(q2)==0):
			self.q1.append(q1)
			self.q2.append(q2)
			self.data_label[self.i]=is_duplicate
		else:
			print q1, q2
		self.i +=1


class SentenceToArrays(object):
	def __init__(self,n,we):
		self.data_x=[]
		self.data_y=[]
		self.data_label=np.zeros(n)
		self.we = we
		self.i=0

	def process(self, q1, q2, is_duplicate):
		x = [self.we.embeddings[self.we.words[w]] for w in q1 if self.we.words.has_key(w)]
		y = [self.we.embeddings[self.we.words[w]] for w in q2 if self.we.words.has_key(w)]
		if not (len(x)==0 or len(y)==0):
			self.data_x.append(x)
			self.data_y.append(y)
			self.data_label[self.i]=is_duplicate
		else:
			print q1, q2
		self.i +=1

class SentenceToArray(object):
	def __init__(self,n,we):
		self.data_x=np.zeros((n,300))
		self.data_y=np.zeros((n,300))
		self.data_label=np.zeros(n)
		self.we = we
		self.i=0

	def process(self, q1, q2, is_duplicate):
		x = [self.we.embeddings[self.we.words[w]] for w in q1 if self.we.words.has_key(w)]
		y = [self.we.embeddings[self.we.words[w]] for w in q2 if self.we.words.has_key(w)]
		if not (len(x)==0 or len(y)==0):
			self.data_x[self.i]=sum(x)/len(x)
			self.data_y[self.i] = sum(y)/len(y)
			self.data_label[self.i]=is_duplicate
		else:
			print q1, q2
		self.i +=1

class WordCounter(object):
	def __init__(self):
		self.wordcounts={}
	
	def process(self, q1, q2, is_duplicate):
		for word in q1+q2:
			if self.wordcounts.has_key(word):
				self.wordcounts[word]+=1
			else:
				self.wordcounts[word]=1
		


class TestQuestions(object):
	def __init__(self, sentence_each):
		with open("data/test.csv", "r") as f:
					reader = csv.reader(f, delimiter=',', quotechar='"')
					next(reader)
					
					i=0
					for line in reader:
						"""
						if i>=max_n:
							break
						if i%1000==0:
							print "Read first ",i," questions"
								
						pair_id = int(line[0])
						qid1 = int(line[1])
						qid2 = int(line[2])
						q1 = line[3]
						q1 = q_to_words(q1)
						#''.join(ch.lower() for ch in q1.replace(","," ") if ch.isalnum() or ch==' ').split()
						q2 = line[4]
						q2 = q_to_words(q2)
						#''.join(ch.lower() for ch in q2.replace(","," ") if ch.isalnum() or ch==' ').split()
						is_duplicate = None
						"""
						i+=1
					print i
						#self.sentence_each.process(q1,q2,is_duplicate)		
	
	
class Questions(object):
	
	def __init__(self, sentence_each):
		self.sentence_each = sentence_each
	
	def load_first_n_questions(self, max_n = 404246):
		self.max_n=max_n

		n_duplicates = 0
		n_total = 0
		
		aggregate=[]
		
		with open("data/train.csv", "r") as f:
			reader = csv.reader(f, delimiter=',', quotechar='"')
			next(reader)
			
			i=0
			for line in reader:
				if i>=max_n:
					break
				if i%1000==0:
					print "Read first ",i," questions"
						
				pair_id = int(line[0])
				qid1 = int(line[1])
				qid2 = int(line[2])
				q1 = line[3]
				q1 = q_to_words(q1)
				#''.join(ch.lower() for ch in q1.replace(","," ") if ch.isalnum() or ch==' ').split()
				q2 = line[4]
				q2 = q_to_words(q2)
				#''.join(ch.lower() for ch in q2.replace(","," ") if ch.isalnum() or ch==' ').split()
				is_duplicate = int(line[5])
				n_total+=1
				n_duplicates+=is_duplicate
				
				self.sentence_each.process(q1,q2,is_duplicate)
				
				i+=1
#				add_synonyms_to_words(q1,words)
#				add_synonyms_to_words(q2,words)

#		embs = load_embeddings()
#		with open ("pickled/filtered_embeddings.pkl","wb") as f:
#			cPickle.dump(filter_embeddings(emb, words), f)
		"""
		with open("x.pkl","wb") as f:
			np.save(f, data_x)
		with open("y.pkl","wb") as f:
			np.save(f, data_y)
		with open("label.pkl", "wb") as f:
			np.save(f, data_label)		
		"""
		
def add_synonyms_to_words(question,words):
	for word in question:
		for synonym in get_all_synonyms(word):
			wordsl.add(synonym)
		wordsl.add(word)
		
def load_data():
	z=None
	with open("pickled/complete_data.pkl", "rb") as f:
		z = np.load(f)
	"""
	data_x = data_y = data_label = None
	with open("x.pkl","rb") as f:
		data_x=np.load(f)
	with open("y.pkl","rb") as f:
		data_y=np.load(f )
	"""

	with open("pickled/label.pkl", "rb") as f:
		data_label=np.load(f)		
	return z,data_label
#	return data_x,data_y,data_label

if __name__=='__main__':
	we = WordEmbeddings()
	we.read_embeddings(5000)
	sta = SentenceToArray(2000,we)
	q = Questions(sta)
	q.load_first_n_questions(2000)
	
	print sta.data_x[:10]
