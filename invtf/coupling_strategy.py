import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np

class CouplingStrategy(): 

	def split(self, X): raise NotImplementedError()

	def combine(self, x0, x1): raise NotImplementedError()

	def coupling_shape(self, input_shape): raise NotImplementedError()

"""
	The coupling layers require input tensor to be split. 
	This strategy splits on the channels. 
	Compared to downscaling image this is substantially simpler.
	This strategy is adopted by Glow [1]. 

	After implementing all models I'll experimentally compare all the 
	different strategies. I think the main argument for this implementation
	is ease of implementation, downscaling might work better. 

	[1] https://github.com/openai/glow/blob/eaff2177693a5d84a1cf8ae19e8e0441715b82f8/model.py#L387
"""
class SplitChannelsStrategy(CouplingStrategy):  


	def split(self, X): 
		c = X.shape[-1]
		x0 = X[:, :, :, :c//2]
		x1 = X[:, :, :, c//2:]
		return x0, x1

	def combine(self, x0, x1): 
		return tf.concat((x0, x1), axis=-1)

	def coupling_shape(self, input_shape): 
		h, w, c = input_shape
		return (h, w, c//2)



"""
	Instead of splitting on channels, one could let the coupling modify all channels
	but a "down sampled" version of each channel. A simple way of doing this is to 
	split on odd/even horizontal pixels (an alternative approach would be vertical pixels). 

	One potential advantage of this approach compared to channel splitting is that images
	have RGB 3 channels, the channel splitting thus usually has a single Squeeze first so
	there are 12 channels which one then splits on. 
		It seems color information (initially distributed on channels) is less important
	than spatial information. The implementation below assumes no Squeeze and no MultiScale. 

	TODO:
		Make a downscale strategy that constructs roughly a half sized image \approx 22x23 on 32x32 or 20x19 for 28x28. 

"""
class ConvAlternateStrategy(CouplingStrategy): 

	"""
		Hacky implementation. Permutation can be done with matrix multiplication.
		It is easy to split this can be done easily by indexing. 
		Combining is a bit harder, we can do it by concatenating and then permuting. 
		The last permutation can be done by matrix mult. 
		
	"""
	def __init__(self):
		P = np.zeros((32, 32), dtype=np.int32)

		for i in range(16): 
			P[2*i, i] = 1
			P[2*i+1, i+16] = 1

		self.P = np.linalg.inv(P)
		

	def split(self, X): 
		self.shape  = tf.shape(X)
		x0 			= X[:, :, ::2, :] 
		x1 			= X[:, :, 1::2, :]
		return x0, x1

	def combine(self, x0, x1):  # might be a bit slow, but only used in sampling. 
		# numpy like way, but not allowed in tensorflow. 
		#x = tf.zeros(self.shape)
		#x[:, :, ::2, :] = x0
		#x[:, :, ::2, :] = x1

		# if we concatenate x0, x1 we just need to interleave/permute columns. 
		C = tf.concat((x0, x1), axis=2) 

		C1 = C[:, :, :, 0] @ self.P
		C2 = C[:, :, :, 1] @ self.P
		C3 = C[:, :, :, 2] @ self.P 

		C1 = tf.reshape(C1, (-1, 32, 32, 1))
		C2 = tf.reshape(C2, (-1, 32, 32, 1))
		C3 = tf.reshape(C3, (-1, 32, 32, 1))
	
		C = tf.concat((C1, C2, C3), axis=-1)

		return C
	

class EvenOddStrategy(CouplingStrategy): 

	def split(self, X): 
		self.shape = tf.shape(X)
		x0		= X[:, ::2]
		x1		= X[:, 1::2]
		return x0, x1

	def combine(self, x0, x1): 
		return tf.reshape(tf.stack([x0, x1], axis=-1), self.shape)

class SplitOnHalfStrategy(CouplingStrategy): 
	
	def split(self, X): 
		d = tf.shape(X)[1]
		x0 		= X[:, :d//2]
		x1 		= X[:, d//2:]
		return x0, x1

	def combine(self, x0, x1): 
		return tf.concat((x0, x1), axis=1)


