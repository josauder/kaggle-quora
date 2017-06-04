import numpy as np

class WordEmbeddings(object):
	def __init__(self):
		self.words_didnt_exist = set([])
		
	def read_embeddings(self,n=200000):
		self.words={}
		self.embeddings = np.zeros((n,300),dtype=float)

		with open("data/wiki.en.vec.part") as f:
			i=0
			next(f)
			for line in f:
				if i<n:
					if i%1000==0:
						print i, " embeddings read"
					ind = line.find(" ")
					word = line[:ind]
					vector = np.array([float(x) for x in line[ind+1:].split(" ") if not x.isspace()])
					
					self.embeddings[i]=vector
					self.words[word]=i
					i+=1
				else:
					break
	
	def save_embeddings(self):
		_save_embeddings(self)
		
	def save_filtered_embeddings(self):
		_save_embeddings("pickled/filtered_embeddings.pkl",
						 "pickled/filtered_words.pkl")
		
	def _save_embeddings(self,
						 embeddingsname="pickled/embeddings.pkl",
						 wordsname = "pickled/words.pkl"):
		print "Saving embeddings"
		with open(embeddingsname, "wb") as f:
			np.save(self.embeddings, f)
		print "Saving words"
		with open(wordsname, "wb") as f:
			cPickle.dump(self.words,f)

	def load_embeddings(self):
		_load_embeddings()
		
	def _load_embeddings(self,
						 embeddingsname="pickled/embeddings.pkl",
						 wordsname = "pickled/words.pkl"):
		print "Loading embeddings: ", embeddingsname
		with open(embeddingsname, "rb") as f:
			self.embeddings= cPickle.load(f)
		print "Loaded embeddings"
		print "Loading words-dict", wordsname
		with open(wordsname, "rb") as f:
			self.words= cPickle.load(f)
		print "Loaded words-dict"

	
	def has_word(self, word):
		if self.words.has_key(word):
			return True
		else:
			self.words_didnt_exist.add(word)
		#	print word
			return False

	def load_filtered_embeddings(self):
		_load_embeddings("pickled/filtered_embeddings.pkl",
						 "pickled/filtered_words.pkl")
		
	def filter_embeddings(embeddings, words):
		print "Filtering embeddings"
		wordset = set(words.keys())
		for i,key in enumerate(embeddings.keys()):
			if i%1000:
				print "Checked ", i, " words while filtering"
			if not key in wordset:
				embeddings.pop(key,None)
		print "Done filtering embeddings"
		return embeddings


if __name__=='__main__':
	we = WordEmbeddings()
	we.read_embeddings(10000)
