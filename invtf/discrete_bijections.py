import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np


class NaturalBijection(keras.layers.Layer): 

	"""
		Reduces dimensionality by a factor two by using a discrete bijection. 
		Assumes the input number of channels is even. If the input is 5-bit
		it can be used twice without overflow for 32bit integers, that is,
		
			g.add(Squeeze()) 			# h, w, c -> h//2, w//2, c*4
			g.add(NaturalBijection())  	# h//2, w//2, c*4 -> h//2, w//2, c*2
			g.add(NaturalBijection())  	# h//2, w//2, c*2 -> h//2, w//2, c

	"""

	def call(self, X): 
		## split on even and odd channels. 
		## assumes even number of channels. 


		X = tf.dtypes.cast(X, dtype=tf.int32) # if not it overflows. 

		c = X.shape[-1] # refactor to use SplitStrategy 
		i = X[:, :, :, :c//2]
		j = X[:, :, :, c//2:]

		def f(i, j): return (i+j)*(i+j+1)//2 + j

		dim_red = f(i,j)

		return tf.dtypes.cast(dim_red, dtype=tf.float32) # the rest of the network expects float32. 

	def call_inv(self, X): 

		n = tf.dtypes.cast(X, dtype=tf.int32)
	
		m = tf.dtypes.cast( (np.sqrt(8*n+1)-1)//2 , dtype=tf.int32)

		j = n - m * (m+1)//2
		i = m - j 

		return tf.concat((i, j), axis=-1)
	
	def log_det(self): return 0.


