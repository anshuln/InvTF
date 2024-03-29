import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'

__version__ = "0.0.1"

import tensorflow as tf

import invtf.coupling_strategy
import invtf.datasets
import invtf.dequantize
import invtf.discrete_bijections
import invtf.grow_memory
import invtf.latent
import invtf.layers
import invtf.models
import invtf.override
import invtf.visualize

import tensorflow.keras as keras 
import numpy as np
import matplotlib.pyplot as plt 

print("TF Version: \t", tf.__version__)
print("Eager: \t\t", tf.executing_eagerly())
print("InvTF Version: \t", __version__)
print("-------------------------------\n")


class Generator(keras.Model): 
	"""
		Linear stack of invertible layers for a generative model. 


		----------------------------------------------------------------------------------------

		Example:	(TODO: Add link to Google Colab w/ 'simple.py' and "pip install invtf". )

			> import invtf
			> import tensorflow.keras as keras

			> # Load data
			> X = invtf.datasets.cifar10()
			> input_shape = X.shape[1:]

			> # Model
			> g = invtf.Generator()

			> # Pre-process
			> g.add(invtf.dequantize.UniformDequantize(input_shape=input_shape)) 
			> g.add(invtf.layers.Normalize()) 

			> # Actual model. 
			> g.add(invtf.layers.Squeeze())

			> for _ in range(10): 
			> 	g.add(invtf.layers.ActNorm())
			> 	g.add(invtf.layers.Inv1x1Conv()) 
			> 	g.add(invtf.layers.Conv3DCirc())
			> 	g.add(invtf.layers.AdditiveCouplingReLU()) 
			>    
			>	# If the line below is uncommented, the architecture
			>	# becomes a multi-scale architecture similar to [1,2].
			>	# if i == 2 or i == 5 or i == 7: g.add(invtf.layers.MultiScale())

			> # Prepare model for training and print summary. 
			> g.compile()		# handles maximum likelihood computations. 
			> g.init(X[:1000])	# handles data dependent initialization as used in [2]. 
			> g.summary()

			> g.fit(X)

		It is now possible to use the training model to sample or interpolate between
		training examples. 

		----------------------------------------------------------------------------------------

		Comments:

			(1) The Generator class supports an API similar to that of Keras Sequential [4]. 
			The example above demonstrates how to build a invertible generative model with
			this API. After defining a model, calling "model.fit" automatically trains the model 
			to maximize the likelihood of the data. Notably, the user does not need to handle any 
			likelihood computations, these are handled automatically by the framework. 

			TODO: outline how one writes new layers by implementing call, call_inv and log_det. 

			Even though the Generator class has an API similar to Sequential, it inherits 
			from the Model class. This is caused by the multi-scale architecture. Essentially,
			Sequential only allows all layers to have one output, and the layer that implements
			the multi-sclae architecture has two outputs. 


			(2) Previously, dequantization and data normalization was thought as pre-processing 
			done before the data is given to the model. Both are treated as a part of the 
			model for the following reasons: 

				- Normalization affects the loss function of invertible generative models,
				  in fact, the normalization typically has a very large jacobian determinant. 
				  Having normalization as a part of the function model allows us to 
				  automatically update the loss function. 

				- The way we choose to dequantize discrete data is a modeling assumption. 
				  We highly recommend reading [3] on variational dequantization. 

			We chose Dequantization and Normalization to be Keras Layers for convenience. 

			(3) Implementing the multi-scale architecture. Automatically handling these 
			computations are a bit tedious, and, unfortunately, complicates reading the code. 

		----------------------------------------------------------------------------------------

		Known-Issues:

			(1) Each multi-scale level increases loss substantially. Currently, it is not clear
			if this is caused by bad initialization or an error in the likelihood computations. 

		----------------------------------------------------------------------------------------

		References:

			[1] Density Estimation using Real NVP. 							https://arxiv.org/abs/1605.08803
			[2] Glow: Generative Flow with Invertible 1x1 Convolutions. 	https://arxiv.org/abs/1807.03039
			[3] Flow++: Improving Flow-Based Generative Models with  		https://arxiv.org/abs/1902.00275
				Variational Dequantization and Architecture Design
			[4] The Sequential model API									https://keras.io/models/sequential/


		Todo:
			

	"""

	def __init__(self, latent=latent.Normal()):
		"""
			
		"""
		super(Generator, self).__init__()
		self.latent = latent 

	def predict(self, X, dequantize=True): 
		"""
			Computes the encoding under the generative model of the input samples. 

			Arguments: 

				X:				Input samples, typically Numpy array.  

				dequantize: 	Boolean. This allows the user to disable dequantization. This can be useful 
								when testing the numerical stability of inverse computations, since the 
								dequantization has no inverse. 

			Returns:		
				Numpy array X of encoding and a list Zs containing numpy arrays of intermediate 
				multi-scale outputs. 

		"""
		Zs = [] 

		for layer in self.layers: 

			# Allow deactivating de-quantize. 
			if not dequantize and isinstance(layer, invtf.dequantize.Dequantize): 		continue	

			# Handle multi-scale intermediate output. 
			if isinstance(layer, invtf.layers.MultiScale): 
				X, Z = layer.call(X)
				Zs.append(Z)
				continue

			X = layer.call(X)

		return X, Zs

	def predict_inv(self, X, Zs=None): 
		"""
			Reconstructs encodings (X, Zs) under the generative model. 

			Arguments: 

				X, Zs:			NumPy arrays with different levels of encoding. 

			Returns:		
				Numpy array X of reconstructions. 

		"""
		n = X.shape[0]

		for layer in self.layers[::-1]: 

			if isinstance(layer, invtf.layers.MultiScale): 
				X = layer.call_inv(X, Zs.pop())

			else: 
				X = layer.call_inv(X)

		return np.array(X)




	def loss(self, y_true, y_pred):	 
		"""
			Computes negative log likelihood in bits per dimension. If the model uses Variational 
			Dequantization it incorporates this into the loss function, see equation (12) from [1]. 

			Arguments: 

				y_true:			Dummy variable. It is required by Keras API, see [2]. 
								but the likelihood computations does not use it.
				y_pred:			The encoding of the model as a single numpy array, *NOT* (X, Zs). 

			Returns:		
				The negative log likelihood of the model on data X such that encodings y_pred 
				is the output of predict(X). 


			Comments: 

				It often useful to inspect the different components of the loss function, for example,
				how big is the jacobian term compared to the latent density term. To ensure these
				are normalized with the output of this function, the final loss, the normalization
				(computing as bit per dimensions) happens inside each of these functions separately. 

			
			[1] Flow++: Improving Flow-Based Generative Models with 			https://arxiv.org/pdf/1902.00275
				Variational Dequantization and Architecture Design
			[2] Usage of loss functions											https://keras.io/losses/

		"""
		return self.loss_log_det(y_true, y_pred) + self.loss_log_latent_density(y_true, y_pred) + self.loss_log_var_dequant(y_true, y_pred)


	def compile(self, optimizer=keras.optimizers.Adam(0.001), **kwargs):	
		"""
			Compiles the model to minimize negative log likelihood. 
			The different terms of the loss function is added as metric, 
			this is often useful information during training. 
		"""

		if "loss" in kwargs.keys(): raise Exception("Currently only supports training with maximum likelihood. Please leave loss unspecified. ")

		kwargs['optimizer'] = optimizer
		kwargs['loss'] 		= self.loss 

		def lg_det(y_true, y_pred): 	return self.loss_log_det(y_true, y_pred)
		def lg_latent(y_true, y_pred): 	return self.loss_log_latent_density(y_true, y_pred)
		def lg_perfect(y_true, y_pred): return self.loss_log_latent_density(y_true, self.latent.sample(shape=tf.shape(y_pred))) 
		def lg_vardeqloss(y_true, y_pred): return self.loss_log_var_dequant(y_true, y_pred)

		kwargs['metrics'] = [lg_det, lg_latent, lg_perfect, lg_vardeqloss]

		super(Generator, self).compile(**kwargs)


	def loss_log_det(self, y_true, y_pred): 
		"""
			Computes negative log determinant, the first term of the loss function. 
			It is normalized to be in bits per dimension.

		"""
		d			= tf.cast(tf.reduce_prod(y_pred.shape[1:]), tf.float32)
		norm		= d * np.log(2.) 

		# Divide by /d to get per dimension. 
		# Divide by log(2) to go from log base E (default in tensorflow) to log base 2. 
		# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math/log
		log_det 	= self.log_det() / norm

		return 		- log_det  


	def loss_log_latent_density(self, y_true, y_pred): 
		"""
			Computes log density of encoded data in latent space, the second term of the loss function. 
			It is normalized to be in bits per dimension. 

			Arguments: 

				y_true:			Dummy variable. It is required by Keras API. 
								but the likelihood computations does not use it.
				y_pred:			The encoding of the model as a single numpy array, *NOT* (X, Zs). 

			Returns:		
				Log latent density of the encoded data in bits per dimension averaged over samples. 

			Comments: 

				The average over the mini-batch is done to balance this term with the log_det.
				Alternatively, one could have scaled log_det by the size of the mini-batch. 
			
		"""

		d			= tf.cast(tf.reduce_prod(y_pred.shape[1:]), 		tf.float32)
		norm		= d * np.log(2.) 

		# Divide by /d to get per dimension. 
		# Divide by log(2) to go from log base E (default in tensorflow) to log base 2. 
		# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math/log
		normal 		= self.latent.log_density(y_pred) / norm 
		batch_size 	= tf.cast(tf.shape(y_pred)[0], 	tf.float32)
		normal		= normal / batch_size
		return 		- normal


	def loss_log_var_dequant(self, y_true, y_pred): 
		"""

		"""
		
		vardeqloss = 0
		for layer in self.layers: 
			if isinstance(layer, invtf.dequantize.VariationalDequantize): 
				vardeqloss = layer.loss()
				break

		d			= tf.cast(tf.reduce_prod(y_pred.shape[1:]), 		tf.float32)
		norm		= d * np.log(2.) 
		vardeqloss 	= vardeqloss / norm
		return - vardeqloss




	def log_det(self):	
		logdet = 0.

		for layer in self.layers: 
			if isinstance(layer, tf.keras.layers.InputLayer): 	continue 
			logdet += layer.log_det()
			
		return logdet





	def add(self, layer): 
		"""
			.. 

			Based loosely on source code of keras.models.Sequential. 
		"""

		if len(self._layers) == 0:
			if not hasattr(layer, "_batch_input_shape"): 
				raise Exception("The first layer should include input dimension, e.g. UniformDequantize(input_shape=X.shape[1:]). ")

		if isinstance(layer, 		keras.layers.InputLayer): 
			raise Exception("Don't add an InputLayer, this is the responsibility of the Generator class. ")

		if not isinstance(layer, 	keras.layers.Layer): 
			raise TypeError("The added layer must be an instance of class Layer. Found: " + str(layer))


		self.built = False
		set_inputs = False


		from tensorflow.python.keras import layers as layer_module
		from tensorflow.python.keras.engine import base_layer
		from tensorflow.python.keras.engine import base_layer_utils
		from tensorflow.python.keras.engine import input_layer
		from tensorflow.python.keras.engine import training
		from tensorflow.python.keras.engine import training_utils
		from tensorflow.python.keras.utils import layer_utils
		from tensorflow.python.keras.utils import tf_utils
		from tensorflow.python.platform import tf_logging as logging
		from tensorflow.python.training.tracking import base as trackable
		from tensorflow.python.util import nest
		from tensorflow.python.util import tf_inspect
		from tensorflow.python.util.tf_export import keras_export



		if not self._layers:	 # if list is empty. 
			#shape 	= layer._batch_input_shape 
			#x 		= keras.layers.InputLayer(input_shape=shape)
			#self._layers.append(x)

			#layer(x)
			#self._layers.append(layer) 

			batch_shape, dtype = training_utils.get_input_shape_and_dtype(layer)
			if batch_shape:
					# Instantiate an input layer.
					x = input_layer.Input(
							batch_shape=batch_shape, dtype=dtype, name=layer.name + '_input')
					# This will build the current layer
					# and create the node connecting the current layer
					# to the input layer we just created.
					layer(x)
					set_inputs = True

			if set_inputs:
				# If an input layer (placeholder) is available.
				if len(nest.flatten(layer._inbound_nodes[-1].output_tensors)) != 1:
					raise ValueError('All layers in a Sequential model '
													 'should have a single output tensor. '
													 'For multi-output layers, '
													 'use the functional API.')
				self.outputs = [
						nest.flatten(layer._inbound_nodes[-1].output_tensors)[0]
				]
				self.inputs = layer_utils.get_source_inputs(self.outputs[0])

		elif self.outputs:
			# If the model is being built continuously on top of an input layer:
			# refresh its output.
			output_tensor = layer(self.outputs[0])
	
			# MAIN NEW LINE
			if isinstance(layer, invtf.layers.MultiScale): output_tensor = output_tensor[0]

			if len(nest.flatten(output_tensor)) != 1:
				raise TypeError('All layers in a Sequential model '
												'should have a single output tensor. '
												'For multi-output layers, '
												'use the functional API.')
			self.outputs = [output_tensor]

		if self.outputs:
			# True if set_inputs or self._is_graph_network or if adding a layer
			# to an already built deferred seq model.
			self.built = True

		if set_inputs or self._is_graph_network:
			self._init_graph_network(self.inputs, self.outputs, name=self.name)
		else:
			self._layers.append(layer)
		if self._layers:
			self._track_layers(self._layers)

		self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)




	def fit(self, X, **kwargs): 
		"""

			TODO:
				maybe remove the dummy X somehow?
		"""
		return super(Generator, self).fit(X, y=X, **kwargs)	# if user specifies batch_size here, get upset. 

	def rec(self, X): 
		X, Zs 	= self.predict(X, dequantize=False) # TODO: deactivate dequantize. 
		rec 	= self.predict_inv(X, Zs)
		return rec

	def check_inv(self, X, precision=10**(-5)): 
		img_shape = X.shape[1:]

		rec = self.rec(X)

		if not np.allclose(X, rec, atol=precision):
			fig, ax = plt.subplots(5, 3)
			for i in range(5): 
				ax[i, 0].imshow(X[i].reshape(img_shape).astype(np.int32))
				ax[i, 0].set_title("Image")
				ax[i, 1].imshow(rec[i].reshape(img_shape))
				ax[i, 1].set_title("Reconstruction")
				ax[i, 2].imshow((X[i]-rec[i]).reshape(img_shape))
				ax[i, 2].set_title("Difference")
				plt.show()


	def inspect_log_det(self):	# integrate this with tensorboard. 
		logdet = 0.

		self.ratio = {}
		self.ratio_ = {}

		norm = 32*32*3 * np.log(2)

		for layer in self.layers: 
			val 		= - layer.log_det() / norm
			logdet 		+= val

			key 		= str(type(layer))
			if not key in self.ratio.keys(): self.ratio[key] = 0
			self.ratio[key] += val

		for key in self.ratio.keys():
			self.ratio_[key] = self.ratio[key] / logdet

		#for key in self.ratio.keys():
		#	print(self.ratio_[key], "\t", self.ratio[key], "\t", key)




	def sample(self, n=1000, fix_latent=True, std=1.0):	
		#Z 	= self.latent.sample(n=n, fix_latent=fix_latent)
		
		# Figure out how to handle shape of Z. If no multi-scale arch we want to do reshape below. 
		# If multi-scale arch we don't want to, predict_inv handles it. Figure out who has the responsibility. 

		output_shape 	= self.layers[-1].output_shape[1:]

		X = self.latent.sample(shape=(n, ) + output_shape, std=std)

		for layer in self.layers[::-1]: 

			if isinstance(layer, invtf.layers.MultiScale): 
				Z = self.latent.sample(shape=X.shape)
				X = layer.call_inv(X, Z)
			else: 
				X = layer.call_inv(X)

		return np.array(X, dtype=np.int32) # makes it easier on matplotlib. 

		return fakes



	def check_init(self, X):	# somethign fishy is going on here. 
		fig, ax = plt.subplots(3, 1)
		img_shape = X.shape[1:]
		fig.canvas.manager.window.wm_geometry("+2500+0")

		fake = self.sample(1, fix_latent=True)

		ax[0].imshow(fake.reshape(img_shape)/255)
		ax[0].set_title("Fake")

		ax[1].imshow(X[0].reshape(img_shape)/255)
		ax[1].set_title("Real")

		ax[2].imshow(self.rec(X[:1]).reshape(img_shape)/255)
		ax[2].set_title("Reconstruction")
		plt.tight_layout()

	def init(self, X):	 # TODO: consider changing this to build and call super(..).build(..) in end?
		"""
			Initial prediction handles some internal keras initialization. 

			After this we add extra initialization which computes normalization
			and data dependent actnorm initialization. These computations are 
			handled in numpy, doing it in tf with GPU would speed up. 
			
			It could also be interleaved with the initial prediction. 
		"""
		self.predict(X)

		for i, layer in enumerate(self.layers): 
			if isinstance(layer, invtf.layers.Normalize): break
			X = layer.call(X)
		
		max = np.max(X.numpy()) 
		layer.scale = 1 / (max / 2)
		X = layer.call(X).numpy()

		assert np.allclose(X.min(), 0, X.max(), 1)

		# Also compute values of actnorm layers, do on cpu. 

		for layer in self.layers[i:]:

			if isinstance(layer, invtf.layers.ActNorm):	# do normalization
				X = X.numpy()
				# normalize channel-wise to mean=0 and var=1. 
				n, h, w, c = X.shape
				for i in range(c):	# go through each channel. 
					mean = np.mean(X[:, :, :, i])
					X[:, :, :, i] = X[:, :, :, i] - mean

					std	= np.std(X[:, :, :, i])
					X[:, :, :, i] = X[:, :, :, i] / std


					std	= np.std(X[:, :, :, i])
					mean = np.mean(X[:, :, :, i])
					assert np.allclose(std, 1) and np.allclose(mean, 0, atol=10**(-4)), (mean, std)

			X = layer.call(X)

			if isinstance(layer, invtf.layers.MultiScale): X = X[0]
		
