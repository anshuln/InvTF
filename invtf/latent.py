import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np

"""
	TODO:
		1. Add mean to log_density of Normal (see wiki multi-variate-normal and do log of pdf)
		2. Add mean and std to logistic, similarly as above, see wikipedia. 
		3. Compare both to mean=0, std=1 case used in Nice. 

"""


class Normal():  # assume no mean so far. 

	def __init__(self, mean=0, std=1): 
		self.mean 	= mean
		self.std 	= std

	def log_density(self, X): 
		return tf.math.reduce_sum( -1/2 * (X**2/self.std**2 + tf.math.log(2*np.pi*self.std**2)) )

	def sample(self, n=1000): 
		return tf.random.normal((n, 2), self.mean, self.std)


class Logistic(): # see NICE page 5 section 3.4 

	def __init__(self):  pass 

	def log_density(self, X):
		return tf.math.reduce_sum( - tf.math.log(1 + tf.exp(X)) - tf.math.log(1 + tf.exp(-X)) )

	def sample(self, n=1000):	 # it seems this is not in tensorflow 2.0.0 beta
		return np.random.logistic(0, 1, size=(1000, 2)).astype(np.float32)

	
