import invtf 
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from invtf.coupling_strategy import *
from tensorflow.keras.layers import *
from invtf.latent import Normal
from invtf.layers import *


class Dequantize(keras.layers.Layer): 
	"""
		Fitting a continuous density model to discrete data will produce 
		a degenerate solution that places all probability mass on discrete 
		data points.  [1]

		A common solutions "dequantize" the data by e.g. adding uniform
		random noise. 

			y = x + u		where u \sim U(0, 1)

		Training a continuous model 'p' on dequantized y can be interpreted as 
		maximizing a lower bound on log likelihood of a certain other discrete 
		model 'P' on the original discrete data. ([1] section 3.1.1)

		This trick circumvents the degenerate solutions and is implemented in
		UniformDequantize below. 

		Unfortunately, this trick has a substantial issue. It essentially 
		asks the continuous model 'p' to assign uniform probability to 
		unit hypercubes x + [0, 1)^D around data x. This seems very unnatural
		for smooth function approximators like neural networks. 

		To circumvent this issue [1] propose a variational dequantization which
		entails learning the dequantization with (for example) a flow model. 
		Instead of adding u\sim U(0,1) we learn add a learned conditional 
		distribution q(u|x). 
	
		This generalizes the uniform dequantizaton by when q(u|x)=U(0,1). 
		Notably, [1] found the variational dequantization to substantially 
		improve performance. 


		[1] https://arxiv.org/abs/1902.00275

	"""

	def call_inv(self, Z): 	raise NotImplementedError
	def log_det(self): 		raise NotImplementedError
	def call(self, X):  	raise NotImplementedError



class UniformDequantize(Dequantize):  
	"""
		UniformDequantize is one strategy for handling the issue outlined in 
		the documentation of the parent class 'Dequantize'. 

		Issue: Fitting a continuous density model to discrete data will produce 
		a degenerate solution that places all probability mass on discrete 
		data points.  [1]

		A common solutions "dequantize" the data by e.g. adding uniform
		random noise. 

			y = x + u		where u \sim U(0, 1)

		Training a continuous model 'p' on dequantized y can be interpreted as 
		maximizing a lower bound on log likelihood of a certain other discrete 
		model 'P' on the original discrete data. ([1] section 3.1.1)

		[1] https://arxiv.org/abs/1902.00275

	"""
	def call(self, X): 		return X + tf.random.uniform(tf.shape(X), 0, 1)
	def call_inv(self, Z): 	return Z
	def log_det(self): 		return 0.
		

class VariationalDequantize(keras.layers.Layer):  	
	"""

		Example of usage: 
		VariationalDequantize learns a model q(u|x) which is used to dequantize
		the data as 

			y = x + u		where u \sim q(u|x)

		Here q(u|x) is learned as a flow model. 


			g = Generator()
			vardeq = VariationalDequantize()
			vardeq.add( ... )
			vardeq.add( ... )
			vardeq.add( ... )

			g.add(vardeq)


		The problem: 
		Fitting a continuous density model to discrete data will produce 
		a degenerate solution that places all probability mass on discrete 
		data points.  [1]

		A common solution "dequantize" the data by e.g. adding uniform
		random noise. 

			y = x + u		where u \sim U(0, 1)

		Variational dequantization learned a conditional distribution q(u|x). 
		As done in [1] we learned q(u|x) with a flow model. 


		High-level implementation idea:
		This layer implements a flow model by using a list of layers self.layers. 
		One can specify a model by adding layers to this list. The flow model
		is then used for variational dequantization. 



		References: 
		[1] https://arxiv.org/abs/1902.00275


		Potential Improvements: 
			Both Uniform and Variational Dequantization assumes the underlying distribution
			is discrete. I think a more reasonable assumption is that the underlying
			distribution is continuous but goes through a discretization process. 

	"""
	def __init__(self): 
		super(VariationalDequantize, self).__init__(name="var_deq")
		self.layers 			= [] # list that holds all layers. 
		self._is_graph_network 	= False 


	def add(self,layer): 	self.layers.append(layer)

	def build(self, input_shape):
		"""
			Build all the layers of the flow model. 

			To produce summary (like keras.Sequential) we store output shapes
			in layers.output_shape_. This is a slight work around. 
		"""
		_, h, w, c = input_shape


		self.latent 			= invtf.latent.Normal(mean=0, std=1)
		print(self.layers)

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

		eps 	= tf.random.normal(tf.shape(X), 0, 1)  # factorize to allow other distributions. 

		self.precompute_loss(eps)

		#eps		= self.latent.sample(shape=tf.shape(X))
		pred 	= self.call_(eps)

		X 		= X  + pred

		return X

	def call_inv(self, Z):	 return Z 

	def log_det(self): 		 return 0.


	def precompute_loss(self, eps): 
		"""
			The implementation is straight-forward, but understanding why it works
			we need to get into the mathy details of [1]. It probably suffices to read
			page 3-4 of their paper. 

			By using log rules we can rewrite the bound from [1] equations (11-12) to: 

				E_(x\sim P_data) [ \log P_model(x)] 
					> E [ lg p_model(x+ q(e|x)) ] - E[ lg p(e) ] - E[lg | \del q(x|e) / \del e| ^-1]
					> E [ lg p_model(x+ q(e|x)) ] - E[ lg p(e) ] + E[lg | \del q(x|e) / \del e| ]

			The expectation is taken with respect to x\sim P_data and e\sim p for 
			some distribution p (typically isotropic Gaussian). Notice that we
			will optimize the above wrt p_model and q, the E[ lg p(e) ] is thus a
			constant. 

					 E [ lg p_model(x+ q(e|x)) ] + E[lg | \del q(x|e) / \del e| ]

			The first term is the normal training criteria, so we only need to add 
			q(e|x), this is handled in 'call' below. The right term is handled by
			log_det term. 

			[1] https://arxiv.org/abs/1902.00275

		"""

		lgdet = 0
		for layer in self.layers: 
			if isinstance(layer, keras.layers.Reshape): continue 
			lgdet += layer.log_det()

		eps_logp = self.latent.log_density(eps)

		self.precomputed_loss = lgdet - eps_logp 

	def loss(self): 
		return -self.precomputed_loss / 127.5 # divide by normalization. 

	def compute_output_shape(self, input_shape): return input_shape


	
	def summary(self, line_length=None, positions=None, print_fn=None):
		"""
			Printing summaries like keras.Sequential. 
		"""
		print_summary(self, line_length=line_length, positions=positions, print_fn=print_fn) 

	def _check_trainable_weights_consistency(self): return True  # workaround. 



