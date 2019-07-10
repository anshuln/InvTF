import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np

"""
	TODO:
		1. Add mean to log_density of Normal (see wiki multi-variate-normal and do log of pdf)
		2. Add mean and std to logistic, similarly as above, see wikipedia. 
		3. Compare both to mean=0, std=1 case used in Nice. 

"""


class Normal():  # assume no mean so far.  ;;; Can we make this infer dimension somehow?

	def __init__(self, d, mean=0, std=1): 
		self.d		= d
		self.mean 	= mean
		self.std 	= std
		self.latent = None

	def log_density(self, X): 
		# refactor a way such that if we have X.shape != (d, ) we update d? 
		return tf.math.reduce_sum( -1/2 * (X**2/self.std**2 + tf.math.log(2*np.pi*self.std**2)) )

	def sample(self, n=1000, fix_latent=False): 

		if self.latent is None: 	self.latent = np.random.normal(0, 1, (n, self.d)).astype(np.float32)
		elif not fix_latent: 		self.latent = np.random.normal(0, 1, (n, self.d)).astype(np.float32)

		return self.latent[:n]


class Logistic(): # see NICE page 5 section 3.4 

	def __init__(self, d):  
		self.d = d
		self.latent = None

	def log_density(self, X):
		return tf.math.reduce_sum( - tf.math.log(1 + tf.exp(X)) - tf.math.log(1 + tf.exp(-X)) )

	def sample(self, n=1000, fix_latent=False):	 # it seems logistic distribution is not in tensorflow 2.0.0 beta
		if self.latent is None: 	self.latent = np.random.logistic(0, 1, (n, self.d)).astype(np.float32)
		elif not fix_latent: 		self.latent = np.random.logistic(0, 1, (n, self.d)).astype(np.float32)

		return self.latent[:n]

	
