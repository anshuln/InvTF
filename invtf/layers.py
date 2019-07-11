import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
from tensorflow.keras.layers import ReLU
from invtf.override import print_summary
from invtf.coupling_strategy import *

"""
	Known issue with multi-scale architecture. 
	The log-det computations normalizes wrt full dimension. 

"""

class Linear(keras.layers.Layer): 

	def __init__(self, **kwargs): super(Linear, self).__init__(**kwargs)

	def build(self, input_shape): 

		assert len(input_shape) == 2
		_, d = input_shape

		self.W = self.add_weight(shape=(d, d), 	initializer='identity', name="linear_weight")
		self.b = self.add_weight(shape=(d), 	initializer='zero',		name="linear_bias")
		
		super(Linear, self).build(input_shape)
		self.built = True

	def call(self, X): 		return X @ self.W + self.b 

	def call_inv(self, Z):  return (Z - self.b) @ tf.linalg.inv(self.W)

	def jacobian(self):		return self.W

	def log_det(self): 		return tf.math.log(tf.abs(tf.linalg.det(self.jacobian())))

	def compute_output_shape(self, input_shape): 
		self.output_shape = input_shape
		return input_shape


class Affine(keras.layers.Layer): 

	"""
		The exp parameter allows the scaling to be exp(s) \odot X. 
		This cancels out the log in the log_det computations. 
	"""

	def __init__(self, exp=False, **kwargs): 
		self.exp = exp
		super(Affine, self).__init__(**kwargs)

	def build(self, input_shape): 

		#assert len(input_shape) == 2
		d = input_shape[1:]

		self.w = self.add_weight(shape=d, 	initializer='ones', name="affine_scale") 
		self.b = self.add_weight(shape=d, 	initializer='zero', name="affine_bias")

		super(Affine, self).build(input_shape)
		self.built = True

	def call(self, X): 		
		if self.exp: 	return X * tf.exp(self.w) + self.b 
		else: 			return X * self.w 		  + self.b

	def call_inv(self, Z):  
		if self.exp:	return (Z - self.b) / tf.exp(self.w)
		else: 			return (Z - self.b) / self.w

	def jacobian(self):		return self.w

	def eigenvalues(self): 	return self.w

	def log_det(self): 		
		if self.exp: 	return tf.reduce_sum(tf.abs(self.eigenvalues()))
		else: 			return tf.reduce_sum(tf.math.log(tf.abs(self.eigenvalues())))

	def compute_output_shape(self, input_shape): 
		self.output_shape = input_shape
		return input_shape



"""
	For simplicity we vectorize input and apply coupling to even/odd entries. 
	Could also use upper/lower. Refactor this to support specifying the pattern as a parameter. 

	TODO: 
		Potentially refactor so we can add directly to AdditiveCoupling instead of creating 'm'
		by (potentially adding to Sequential) and passing this on to AdditiveCoupling. 
		The main issue is AdditiveCoupling is R^2-> R^2 while m:R^1->R^1, so if we 
		add directly to AdditiveCoupling we run into issues with miss matching dimensions. 
	
"""
class AdditiveCoupling(keras.Sequential): 

	unique_id = 1

	def __init__(self, part=0, strategy=SplitOnHalfStrategy()): # strategy: alternate / split  ;; alternate does odd/even, split has upper/lower. 
		super(AdditiveCoupling, self).__init__(name="add_coupling_%i"%AdditiveCoupling.unique_id)
		AdditiveCoupling.unique_id += 1
		self.part 	= part 
		self.strategy = strategy


	def build(self, input_shape):

		self.layers[0].build(input_shape=(None, 28**2/2))
		out_dim = self.layers[0].compute_output_shape(input_shape=(None, 28**2/2))

		for layer in self.layers[1:]:  
			layer.build(input_shape=out_dim)
			out_dim = layer.compute_output_shape(input_shape=out_dim)

	def call_(self, X): 
		for layer in self.layers: 
			X = layer.call(X)
		return X

	def call(self, X): 		
		shape 	= tf.shape(X)
		d 		= tf.reduce_prod(shape[1:])
		X 		= tf.reshape(X, (shape[0], d))

		x0, x1 = self.strategy.split(X)

		if self.part == 0: x0 		= x0 + self.call_(x1)
		if self.part == 1: x1 		= x1 + self.call_(x0)

		X = self.strategy.combine(x0, x1)

		X 		= tf.reshape(X, shape)
		return X

	def call_inv(self, Z):	 
		shape 	= tf.shape(Z)
		d 		= tf.reduce_prod(shape[1:])
		Z 		= tf.reshape(Z, (shape[0], d))

		z0, z1 = self.strategy.split(Z)
		
		if self.part == 0: z0 		= z0 - self.call_(z1)
		if self.part == 1: z1 		= z1 - self.call_(z0)

		Z = self.strategy.combine(z0, z1)

		Z 		= tf.reshape(Z, shape)
		return Z


	def log_det(self): 		return 0. 

	def compute_output_shape(self, input_shape): return input_shape



"""
	The affine coupling layer is described in NICE, REALNVP and GLOW. 
	The description in Glow use a single network to output scale s and transform t, 
	it seems the description in REALNVP is a bit more general refering to s and t as 
	different functions. From this perspective Glow change the affine layer to have
	weight sharing between s and t. 
	 Specifying a single function is a lot simpler code-wise, we thus use that approach. 


	For now assumes the use of convolutions 

"""
class AffineCoupling(keras.Sequential):  	

	unique_id = 1

	def __init__(self, part=0, strategy=SplitOnHalfStrategy()): 
		super(AffineCoupling, self).__init__(name="aff_coupling_%i"%AffineCoupling.unique_id)
		AffineCoupling.unique_id += 1
		self.part 		= part 
		self.strategy 	= strategy

	def build(self, input_shape):

		# handle the issue with each network output something larger. 
		_, h, w, c = input_shape

		self.layers[0].build(input_shape=(None, h, w//2, c))
		out_dim = self.layers[0].compute_output_shape(input_shape=(None, h, w//2, c))
		self.layers[0].output_shape_ = out_dim

		for layer in self.layers[1:]:  
			layer.build(input_shape=out_dim)
			out_dim = layer.compute_output_shape(input_shape=out_dim)
			layer.output_shape_ = out_dim

	def call_(self, X): 

		in_shape = tf.shape(X)

		for layer in self.layers: 
			X = layer.call(X) # residual 

		# TODO: Could have a part of network learned specifically for s,t to not ONLY have wegith sharing? 
		d = tf.shape(X)[2]
		s = X[:, :, d//2:, :]
		t = X[:, :, :d//2, :]  

		s = tf.reshape(s, in_shape)
		t = tf.reshape(t, in_shape)

		return s, t

	def call(self, X): 		

		x0, x1 = self.strategy.split(X)

		if self.part == 0: 
			s, t 	= self.call_(x1)
			x0 		= x0*s + t

		if self.part == 1: 
			s, t 	= self.call_(x0)
			x1 		= x1*s + t 

		X 		= self.strategy.combine(x0, x1)
		return X

	def call_inv(self, Z):	 
		z0, z1 = self.strategy.split(Z)
		
		if self.part == 0: 
			s, t 	= self.call_(z1)
			z0 		= (z0 - t)/s
		if self.part == 1: 
			s, t 	= self.call_(z0)
			z1 		= (z1 - t)/s

		Z 		= self.strategy.combine(z0, z1)
		return Z


	def log_det(self): 		 

		# TODO: save 's' instead of recomputing. 

		X 		= self.input
		n 		= tf.dtypes.cast(tf.shape(X)[0], tf.float32)

		x0, x1 = self.strategy.split(X)

		if self.part == 0: 
			s, t 	= self.call_(x1)
		if self.part == 1: 
			s, t 	= self.call_(x0)

		# there is an issue with 's' being divided by dimension 'd' later:
		# If we used MultiScale it will be lower dimensional, in this case
		# we should not divide by d but d//2. 

		return tf.reduce_sum(tf.math.log(tf.abs(s))) / n

	def compute_output_shape(self, input_shape): return input_shape

	def summary(self, line_length=None, positions=None, print_fn=None):
		print_summary(self, line_length=line_length, positions=positions, print_fn=print_fn) # fixes stupid issue.




"""
	Try different techniques: I'm implementing the simplest case, just reshape to desired shape. 
	TODO: Implement the following Squeeze strategies: 
		- RealNVP
		- Downscale images, e.g. alternate pixels and have 4 lower dim images and stack them. 
		- ... 
"""
class Squeeze(keras.layers.Layer): 

	def call(self, X): 
		n, self.w, self.h, self.c = X.shape
		return tf.reshape(X, [-1, self.w//2, self.h//2, self.c*4])

	def call_inv(self, X): 
		return tf.reshape(X, [-1, self.w, self.h, self.c])
		
	def log_det(self): return 0. 


# TODO: for now assumes target is +-1, refactor to support any target. 
# Refactor 127.5 
class Normalize(keras.layers.Layer):  # normalizes data after dequantization. 

	def __init__(self, target=[-1,+1], scale=127.5, input_shape=None): 
		super(Normalize, self).__init__(input_shape=input_shape)
		self.target = target
		self.d 		= np.prod(input_shape)
		self.scale  = 1/127.5

	def call(self, X):  
		X 			= X * self.scale  - 1
		return X

	def call_inv(self, Z): 
		Z = Z + 1
		Z = Z / self.scale
		return Z

	def log_det(self): return self.d * tf.math.log(self.scale) 


class MultiScale(keras.layers.Layer): 

	def call(self, X):  # TODO: have different strategies here, and combine it with how coupling layer works? 
		n, w, h, c = X.shape
		Z = X[:, :, :, c//2:]
		X = X[:, :, :, :c//2]
		return X, Z
	
	def call_inv(self, X, Z): 
		return tf.concat((X, Z), axis=-1)

	def compute_output_shape(self, input_shape): 
		n, h, w, c = input_shape
		return (n, h, w, c//2)

	def log_det(self): return 0.




class Inv1x1Conv(keras.layers.Layer):  
	pass # Use decomposition in original article and that of emergin convolutions. 


class InvResNet(keras.layers.Layer): 			pass # model should automatically use gradient checkpointing if this is used. 


# the 3D case, refactor to make it into the general case. 
# make experiment with nD case, maybe put reshape into it? 
# Theoretically time is the same? 
class CircularConv(keras.layers.Layer): 

	def __init__(self, dim=3):  # 
		self.dim = dim 

	def call(self, X): 		pass
	
	def call_inv(self, X): 	pass

	def log_det(self): 		pass


