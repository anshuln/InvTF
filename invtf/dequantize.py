import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import invtf 
from invtf.coupling_strategy import *
from tensorflow.keras.layers import *
from invtf.latent import Normal
from invtf.layers import *

class UniformDequantize(keras.layers.Layer):  

	def __init__(self, input_shape=None, amount=1.0): 
		self.amount = amount
		self.d		= input_shape
		super(UniformDequantize, self).__init__(input_shape=input_shape)#input_shape)

	def call(self, X): 
		return X + tf.random.uniform(tf.shape(X), 0, 1)

	def call_inv(self, Z): return Z
		
	def log_det(self): return 0.
		

class VariationalDequantize(keras.layers.Layer):  

	"""

	"""

	def __init__(self, input_shape=None, dequantize_model=None): 
		#self.input_shape		= input_shape
		self.dequantize_model 	= dequantize_model

		d = np.prod(input_shape)

		#self.noise 				= Normal(d) # refactor later. 

		if self.dequantize_model is None: 

			self.dequantize_model = invtf.Generator()
			self.dequantize_model.add(Squeeze(input_shape=input_shape, name="AKLSDJASKLDJAKL"))

			ac = AffineCoupling(part=0, strategy=SplitChannelsStrategy())
			ac.add(Flatten())
			ac.add(Dense(50, activation="relu"))
			ac.add(Dense(d, bias_initializer="ones", kernel_initializer="zeros"))

			self.dequantize_model.add(ac)

			"""ac = AffineCoupling(part=1, strategy=SplitChannelsStrategy())
			ac.add(Flatten())
			ac.add(Dense(50, activation="relu"))
			ac.add(Dense(d, bias_initializer="ones", kernel_initializer="zeros"))

			self.dequantize_model.add(ac)"""
			self.dequantize_model.add(Reshape(input_shape))

			dummy = tf.random.normal( (2, 32,32,3), 0, 1)
			self.dequantize_model.predict(dummy)
			self.dequantize_model.summary()
			for layer in self.dequantize_model.layers:
				if isinstance(layer, AffineCoupling): layer.summary()

			print("\n"*5)

		super(VariationalDequantize, self).__init__(input_shape=input_shape)

	def call(self, X): 
		# TODO: Make conditional, i.e., make it depend on X somehow. 
		epsilon 	= tf.random.normal(tf.shape(X), 0, 1)
		pred 		= self.dequantize_model.call(epsilon) 
		return X + pred

	def call_inv(self, Z): return Z
		
	# ignoring the p(eps) term since it is constant, see [1] (flow++ page 4-5). 
	def log_det(self): return - self.dequantize_model.log_det()
