import numpy as np
from parse import *


if __name__ == '__main__':

	we = WordEmbeddings()
	we.read_embeddings(300000)
	weighter = Weighter(we,2345796)
	
	q = Questions(weighter)
	q.load_first_n_questions()
	TestQuestions(None)
