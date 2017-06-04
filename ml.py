from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE as PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from word_embeddings import *
from copy import copy
from parse import *
from parse2 import *
import time

"""
we = WordEmbeddings()
we.read_embeddings(100000)


pca = PCA(n_components=3)
d = pca.fit_transform(we.embeddings)
fig = plt.figure()
ax = fig.gca(projection='3d')

def plot_word(word,we):
	_plot_ax(d[we.words[word]], word)
	
def _plot_ax(coords,string):
	a,b,c = coords
	try:
		ax.scatter(a,b,c,color='red')
		ax.text(a,b,c,string, size =10, zorder =1)
		plt.pause(0.01)
		ax.draw()
	except Exception as e:
			print e

def plot_words(strings_list, we):
	strings_list = [x for x in strings_list if we.has_word(x) and x]
	coords_list = [d[we.words[w]] for w in strings_list]
	coords = coords_list[0]
	string = strings_list[0]
	old_coords = None

	l = len(strings_list)
	avg = np.sum(np.array(coords_list),axis = 0) / l
	for i in range(1,l):
		old_coords = copy(coords)
		coords *= i
		coords += coords_list[i]
		coords /= float(i+1)
		string+=strings_list[i]+"-"
#		print old_coords - coords
		ax.plot(np.array([old_coords[0],coords[0]])-avg[0],
				np.array([old_coords[1],coords[1]])-avg[1],
				np.array([old_coords[2],coords[2]])-avg[2],
				color = (1-float(i)/(l-1),0,float(i)/(l-1)),
				alpha=0.5)
#		ax.text(,string, size =10, zorder =1)

def plot_coords(coords_list, we):
	coords = coords_list[0]
	
	
	old_coords = None
	l = len(coords_list)
#	avg = sum(np.array(coords_list))/l
#	print avg
	for i in range(1,l):
		old_coords = copy(coords)
		coords *= i
		coords -= coords_list[i]
		coords /= float(i+1)
		ax.plot(np.array([old_coords[0],coords[0]]),#-a,
				np.array([old_coords[1],coords[1]]),#-b,
				np.array([old_coords[2],coords[2]]),#-c,
				color = (1-float(i)/(l-1),0,float(i)/(l-1)),
				alpha=0.5)
				
#		ax.text(,string, size =10, zorder =1)

#plot_words(["how", "are", "these", "things", "related"],we)
#plot_words(["why", "are", "two", "possible", "things", "connected"],we)
#plt.show()

#we = WordEmbeddings()
#we.read_embeddings(100000)
sta = SentenceToWords(2000,we)
q = Questions(sta)
q.load_first_n_questions(2000)

for i in range(200):
	print "checking", i
	if sta.data_label[i]:
		print i, "is same"
		x,y = sta.q1[i], sta.q2[i]
		plot_words(x,we)
		plot_words(y,we)


print len(we.words_didnt_exist)
"""
"""x,y,l = load_data()

l = l.astype(int)

z =(x-y)
#z = z / (0.01+z.sum(axis=1)[:, np.newaxis])
with open("complete_data","wb") as f:
	np.save(f,z)

"""
"""import time


begin = time.time()
z, l = load_data()
t = time.time() - begin
print "loaded data in", t

l = l.astype(int)
"""

"""
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
pca.fit(z)
d = pca.transform(z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1000):
	a,b,c = d[i]
	if l[i]==1:
		ax.scatter(a,b,c,color='red')
	else:
		ax.scatter(a,b,c,color='blue')

plt.show()
"""
"""
wc = WordCounter()
q = Questions(wc)
q.load_first_n_questions()
"""


def main():
	
	we = WordEmbeddings()
	we.read_embeddings(500000)
	weighter = Weighter(we)
	
	q = Questions(weighter)
	q.load_first_n_questions()

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

	#train = Dataset(z[full], la[full])
	#test  = Dataset(z[testindices], la[testindices])

	train = Dataset(z,la)
	#######
	## EVALUATE
	#######

	#print test.images[:10]
	#print test.labels[:100]

	#train = Dataset(z,la)
	#test = 

	net = network2.Network([1200,600,300,100,2])
	
	net.SGD(train, 20000, 32, 1, lmbda = 0.0005, keep_prob=0.5, save=True)
	#net.SGD(train, 20000, 32, 100, test_data=test, lmbda = 0.0005, keep_prob=0.5)

if __name__=='__main__':
	main()
