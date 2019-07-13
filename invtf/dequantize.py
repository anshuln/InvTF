import invtf 
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from invtf.coupling_strategy import *
from tensorflow.keras.layers import *
from invtf.latent import Normal
from invtf.layers import *

class UniformDequantize(keras.layers.Layer):  

	"""


	"""

	def __init__(self, input_shape=None, amount=1.0): 
		self.amount = amount
		self.d		= input_shape
		super(UniformDequantize, self).__init__(input_shape=input_shape)#input_shape)

	def call(self, X): 
		return X + tf.random.uniform(tf.shape(X), 0, 1)

	def call_inv(self, Z): return Z
		
	def log_det(self): return 0.
		

class VariationalDequantize(keras.layers.Layer):  	


	def add(self,layer): self.layers.append(layer)

	def __init__(self): 
		super(VariationalDequantize, self).__init__(name="var_deq")
		self.part 		= 0
		self.strategy 	= SplitChannelsStrategy()
		self.layers=[]
		self._is_graph_network = False

	def _check_trainable_weights_consistency(self): return True

	def build(self, input_shape):

		# handle the issue with each network output something larger. 
		_, h, w, c = input_shape

		self.layers[0].build(input_shape=(None, h, w, c))
		out_dim = self.layers[0].compute_output_shape(input_shape=(None, h, w, c))
		self.layers[0].output_shape_ = out_dim

		for layer in self.layers[1:]:  
			layer.build(input_shape=out_dim)
			out_dim = layer.compute_output_shape(input_shape=out_dim)
			layer.output_shape_ = out_dim

		super(VariationalDequantize, self).build(input_shape=input_shape)
		self.built = True

	def call_(self, X): 
		for layer in self.layers: 
			X = layer.call(X) 

		return X

	def call(self, X): 		

		eps = tf.random.normal(tf.shape(X), 0, 1)
		pred = self.call_(eps)

		X 		= X  + pred

		return X

	def call_inv(self, Z):	 
		return Z # maybe remove dequantize thing? 



	"""
		Fix these computations and explain. 
	"""
	def log_det(self): 		 

		lgdet = 0
		for layer in self.layers: 
			if isinstance(layer, keras.layers.Reshape): continue 
			print(layer)
			lgdet += layer.log_det()

		return 0. # -lgdet  # see flow ++ page 4-5 , the other term is a constant. 

	def compute_output_shape(self, input_shape): return input_shape

	def summary(self, line_length=None, positions=None, print_fn=None):
		print_summary(self, line_length=line_length, positions=positions, print_fn=print_fn) # fixes stupid issue.







