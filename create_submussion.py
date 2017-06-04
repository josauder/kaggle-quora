import numpy as np
from parse import *

import time

we = WordEmbeddings()
we.read_embeddings(10000)
weighter = Weighter(we,10000)

q = Questions(weighter)
q.load_first_n_questions(10000)


z, l = weighter.arr, weighter.is_duplicate

weighter = None
we = None
q = None

######
## TEST
######
begin = time.time()
network2.shuffle_same_indices(z,l)
t = time.time() - begin
print "shuffled in", t

ones = np.argwhere( l == 1)
zeros = np.argwhere ( l == 0)

begin = time.time()
la = np.zeros((l.size, 2))
la[ones] = np.array([0,1])
la[zeros] = np.array([1,0])
t = time.time() - begin
print "converted labels to one-hot encoded in", t
	
ones = ones [:2000]
zeros = zeros [:2000]

full = np.ones(l.shape, dtype=int)
full[ones] = 0
full[zeros] = 0
full = np.argwhere(full)[:,0].reshape(l.shape[0]-4000)

testindices = np.zeros(l.shape, dtype = int)
testindices[ones]=1
testindices[zeros]=1
testindices = np.argwhere(testindices)[:,0].reshape(4000)

train = Dataset(z[full], la[full])
test  = Dataset(z[testindices], la[testindices])



net = network2.Network([1200,100,2])
sess = net.restore("my-model-10")
print net.get_results(test,sess)
sess.close()



#######
## EVALUATE
#######

#train = Dataset(z,la)
#test = 

