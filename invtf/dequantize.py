import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class UniformDequantize(keras.layers.Layer):  

	def __init__(self, input_shape=None, amount=1.0): 
		self.amount = amount
		self.d		= input_shape
		super(UniformDequantize, self).__init__(input_shape=input_shape)#input_shape)

	def call(self, X): 
		return X + tf.random.uniform(tf.shape(X), 0, 1)

	def call_inv(self, Z): return Z
		
	def log_det(self): return 0.
		

class VariationalDequantize(keras.layers.Layer):  pass  # maybe have model; learn this thing like in flow++ 

