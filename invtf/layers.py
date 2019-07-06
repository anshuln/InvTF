import tensorflow as tf
import tensorflow.keras as keras 

class Linear(keras.layers.Layer): 

	def build(self, input_shape): 

		assert len(input_shape) == 2
		print(input_shape)
		_, d = input_shape

		self.W = self.add_weight(shape=(d, d), 	initializer='identity')
		self.b = self.add_weight(shape=(d), 	initializer='zero')

	def call(self, X): 		return X @ self.W + self.b 


	def call_inv(self, Z):  return (Z - self.b) @ tf.linalg.inv(self.W)

	def jacobian(self):		return self.W

	def log_det(self): 		return tf.math.log(tf.abs(tf.linalg.det(self.jacobian())))



class Affine(keras.layers.Layer): 

	def build(self, input_shape): 

		assert len(input_shape) == 2
		print(input_shape)
		_, d = input_shape

		self.w = self.add_weight(shape=(d), 	initializer='ones') 
		self.b = self.add_weight(shape=(d), 	initializer='zero')

	def call(self, X): 		return X * self.w + self.b 


	def call_inv(self, Z):  return (Z - self.b) / self.w

	def jacobian(self):		return self.w

	def eigenvalues(self): 	return self.w

	def log_det(self): 		return tf.reduce_sum(tf.math.log(tf.abs(self.eigenvalues())))



# invertible non-linearity; coupling layer with ReLU inside. 
# for now coupling layer just splits on last dimension for simplicity
# the last dimension is assumed to be even. 
class CoupledReLU(keras.layers.Layer): 

	def call(self, X):  		pass 
		
	def call_inv(self, Z):  	pass 

	def log_det(self): 			return 0

class Inv1x1Conv(keras.layers.Layer):  pass 

class VarDequantize(keras.layers.Layer):  		pass 

class UniformDequantize(keras.layers.Layer): 	pass 

class AdditiveCoupling(keras.layers.Layer):  	pass 

class AffineCoupling(keras.layers.Layer):  		pass 


class InvResNet(keras.layers.Layer): 			pass # model should automatically use gradient checkpointing if this is used. 

	 
