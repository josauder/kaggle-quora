"""
network.py
~~~~~~~~~~

A simple Tensorflow implementation of Michael Nielsen's `network.py`
from the excellent online-book 'Neural Networks and Deep Learning'.
The goal is not to write eloquent Tensorflow code, but to keep structure
similar to Michael's implementation, which can be found at:
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
"""

#### Libraries
# Standard library
import random

# Third-party library
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def QuadraticCost(x,y):
	return tf.reduce_mean(tf.square(x-y))

def CrossEntropyCost(x,y):
	#1 Note that this deviates from Michael's implementation as a softmax layer is also added
	return tf.nn.softmax_cross_entropy_with_logits(logits=x,labels=y)

class Network(object):	
	
	def __init__(self, sizes,
			#1 cost
			cost=CrossEntropyCost):
		"""Use just like in 'Neural Networks and Deep Learning'"""
		self.num_layers = len(sizes)
		self.sizes = sizes
		
		# Placeholders for inputs that will be fed
		self.input_layer = tf.placeholder("float32", shape=[None, sizes[0]])
		self.correct_labels = tf.placeholder("float32", shape=[None, sizes[-1]])
		self.learning_rate = tf.placeholder("float32", shape=[])
		self.keep_prob = tf.placeholder(tf.float32)

		#3 regularizer
		self.lmbda = tf.placeholder("float32",shape=[])
		
		#2 default_weight_initializer
		self.biases = [tf.Variable(tf.random_normal([size], stddev=0.1, mean = 0.5)) for size in self.sizes[1:]]
		self.weights = [tf.Variable(tf.random_normal([x,y], stddev=0.1, mean = 0.5)/np.sqrt(x)) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
		
		#3 regularizer
		self.regularizers = [tf.nn.l2_loss(t) for t in self.weights]
		
		# Connect fully connected layers
		current_layer = self.input_layer
		for i in range(self.num_layers - 1):
			current_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(current_layer, self.weights[i]) + self.biases[i]), self.keep_prob)
		
		self.output_layer = tf.nn.softmax(current_layer)
		#1
		#self.cost_value = cost(current_layer, self.correct_labels)
		
		#3
		self.cost_value = tf.reduce_mean(cost(current_layer, self.correct_labels) + self.lmbda * sum(self.regularizers))
		
		self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost_value)
		
		# Compare and count correctly predicted images
		correct_prediction = tf.equal(tf.argmax(current_layer,1), tf.argmax(self.correct_labels,1))
		self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
#		self.log_loss = tf.losses.log_loss(labels=self.correct_labels + 10e-15, predictions = current_layer)
	
	def evaluate(self, test_data, sess):
		return self.accuracy.eval(
			session=sess,
			feed_dict={
				self.input_layer: test_data.images, 
				self.correct_labels: test_data.labels,
				self.keep_prob:1})
				
	def get_results(self, test_data, sess):
		return self.output_layer.eval(
			session=sess,
			feed_dict={
				self.input_layer: test_data.images, 
				self.keep_prob:1})

	
	
	def restore(self, checkpoint, test_data):
		with tf.Session() as sess:
			saver.restore(sess, checkpoint)
			
			print get_results(test_data, sess)

			

	"""	def evaluate_log_loss(self, test_data, sess):
		return self.log_loss.eval(
					session=sess,
					feed_dict={
					self.input_layer: test_data.images,
					self.correct_labels: test_data.labels})"""
		
	def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, 
			keep_prob=1, test_data=None, save=False):
		"""Stochastic gradient descent training, usage just like in 
		Michael's implementation"""
		if test_data: n_test = len(test_data.images)
		n = len(training_data.images)
		
		sess=tf.Session()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep=3)

		for j in xrange(epochs):
			shuffle_same_indices(training_data.images,training_data.labels)
			
			for k in xrange(0,n,mini_batch_size):
				image_batch = training_data.images[k:k+mini_batch_size]
				label_batch = training_data.labels[k:k+mini_batch_size]

				sess.run(self.train_step,
					feed_dict={	
						self.input_layer: image_batch, 
						self.correct_labels: label_batch,	
						self.learning_rate: eta,
						self.lmbda: lmbda,
						self.keep_prob : keep_prob})
			if j%10==0 and j!=0:
				eta =eta*0.97
				print "learning rate adapted to: ", eta
			
			if test_data and not save:
				print("Epoch {0}: {1} / {2}"#, log_loss: {3}"
					.format(j,self.evaluate(test_data,sess),n_test))#, self.evaluate_log_loss(test_data,sess)))
			if j%10==0 and j!=0 and save:
				saver.save(sess, "my-model", global_step=j)	
			
			
			#if j%4==0 and j!=0 and test_data and save:
			#	print "saving output to disk"
			#	with open("/home/jonathan/workspace/quora/output", "wb") as f:
			#		np.save(f,self.get_results(test_data, sess)[:,1])
				
			else:
				print("Epoch {0} complete".format(j))
		sess.close()

#### Miscellaneous functions
def shuffle_same_indices(a, b):
	"""Used to shuffle two arrays in unison, as our MNIST-data structure
	is looks different than in 'Neural Networks and Deep Learing'"""
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)

if __name__ == '__main__':
	net = Network([784,100,30,10])
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	net.SGD(mnist.train, 5, 10, 0.015, test_data=mnist.test,lmbda = 0.0005, keep_prob = 0.8)
