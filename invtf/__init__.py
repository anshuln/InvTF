"""
	Explain file. 

"""

import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np


# Assume normal distribution for now. 
std_dev = 1 
def log_normal_density(x): return tf.math.reduce_sum( -1/2 * (x**2/std_dev**2 + tf.math.log(2*np.pi*std_dev**2)) )

"""

	TODO: 

	- Support specifying different latent distributions, see e.g. NICE. 

	- The fit currently uses a dummy 'y=X'. It is not used, but removing it causes an error with 'total_loss'. 
	  Removing might speed up. 

"""
class Generator(keras.Sequential): 

		
	def predict_inv(self, Z): 

		for layer in self.layers[::-1]: 
			Z = layer.call_inv(Z)

		return Z

	def log_det(self): 
		logdet = 0.

		for layer in self.layers: 
			if isinstance(layer, tf.keras.layers.InputLayer): 	continue 
			logdet += layer.log_det()
		return logdet


	def loss(self, y_true, y_pred):   
		#  computes negative log likelihood in bits per dimension. 
		return self.loss_log_det(y_true, y_pred) + self.loss_log_normal_density(y_true, y_pred)

	def loss_log_det(self, y_true, y_pred): 
		# divide by /d to get per dimension and divide by log(2) to get from log base E to log base 2. 
		d			= tf.cast(y_pred.shape[1], 		tf.float32)
		norm		= d * np.log(2.) 
		log_det 	= self.log_det() / norm

		return 		- log_det


	def loss_log_normal_density(self, y_true, y_pred): 
		# divide by /d to get per dimension and divide by log(2) to get from log base E to log base 2. 
		batch_size 	= tf.cast(tf.shape(y_pred)[0], 	tf.float32)
		d			= tf.cast(y_pred.shape[1], 		tf.float32)
		norm		= d * np.log(2.) 
		normal 		= log_normal_density(y_pred) / (norm * batch_size)

		return 		- normal

	def compile(self, **kwargs): 
		kwargs['loss'] 		= self.loss # overrides what'ever loss the user specifieds; change to complain with exception if they specify it with

		def lg_det(y_true, y_pred): 	return self.loss_log_det(y_true, y_pred)
		def lg_latent(y_true, y_pred): 	return self.loss_log_normal_density(y_true, y_pred)
		def lg_perfect(y_true, y_pred): return self.loss_log_normal_density(y_true, np.random.normal(0,1, size=(1000, 2)))

		kwargs['metrics'] = [lg_det, lg_latent, lg_perfect]

		super(Generator, self).compile(**kwargs)

	def fit(self, X, **kwargs): return super(Generator, self).fit(X, y=X, **kwargs)  # if user specifies batch_size here, get upset. 


	def check_inv(self, X): 
		enc = self.predict(X)
		dec = self.predict_inv(enc)
		assert np.allclose(X, dec.numpy(), atol=10**(-5))


		
