"""
	Explain file. 

"""

import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'

import tensorflow as tf
import invtf.grow_memory
import tensorflow.keras as keras 
import numpy as np
import invtf.latent
import matplotlib.pyplot as plt 
from invtf.dequantize import *
from invtf.layers import *



"""

	TODO: 

	- Support specifying different latent distributions, see e.g. NICE. 

	- The fit currently uses a dummy 'y=X'. It is not used, but removing it causes an error with 'total_loss'. 
		Removing might speed up. 

	Comments:
		We are miss-using the Sequential thing as it is normally just a linear stack of layers. 
		If we use the multi-scale architecture this is not the case, as it has multiple outputs. 

"""
class Generator(keras.Sequential): 

	def __init__(self, latent=latent.Normal(28**2)):
		self.latent = latent 

		super(Generator, self).__init__()



	# Sequential is normally only for linear stack, however, the multiple outputs in multi-scale architecture
	# is fairly straight forward, so we change Sequential slightly to allow multiple outputs just for the
	# case of the MultiScale layer. Refactor this to make a new variant MutliSqualeSequential which
	# Generator inherents from. 
	
	def add(self, layer): 
		from tensorflow.python.keras.utils import tf_utils
		from tensorflow.python.keras.engine import training_utils
		from tensorflow.python.util import nest
		from tensorflow.python.keras.utils import layer_utils
		from tensorflow.python.util import tf_inspect


		# If we are passed a Keras tensor created by keras.Input(), we can extract
		# the input layer from its keras history and use that without any loss of
		# generality.
		if hasattr(layer, '_keras_history'):
			origin_layer = layer._keras_history[0]
			if isinstance(origin_layer, keras.layers.InputLayer):
				layer = origin_layer

		if not isinstance(layer, keras.layers.Layer):
			raise TypeError('The added layer must be '
											'an instance of class Layer. '
											'Found: ' + str(layer))

		tf_utils.assert_no_legacy_layers([layer])

		self.built = False
		set_inputs = False
		if not self._layers:
			if isinstance(layer, keras.layers.InputLayer):
				# Corner case where the user passes an InputLayer layer via `add`.
				assert len(nest.flatten(layer._inbound_nodes[-1].output_tensors)) == 1
				set_inputs = True
			else:
				batch_shape, dtype = training_utils.get_input_shape_and_dtype(layer)
				if batch_shape:
					# Instantiate an input layer.
					x = keras.layers.Input(
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
			if len(nest.flatten(output_tensor)) != 1 and not isinstance(layer, MultiScale):
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



	def predict(self, X, dequantize=True): 

		Zs = [] 

		for layer in self.layers: 

			# allow deactivating dequenatize 
			# refactor to just look into name of layer and skip if it has dequantize in name or something like that. 
			if not dequantize and isinstance(layer, UniformDequantize): continue	

			if isinstance(layer, MultiScale): 
				X, Z = layer.call(X)
				Zs.append(Z)
				continue

			X = layer.call(X)

		# TODO: make sure this does not break case without multiscale architecture.
		# append Zs to X;; do by vectorize and then concat. 

		return X, Zs

	def predict_inv(self, X, Z=None): 
		n = X.shape[0]

		for layer in self.layers[::-1]: 
			print(layer, X.shape)

			if isinstance(layer, MultiScale): 
				X = layer.call_inv(X, Z.pop())

			else: 
				X = layer.call_inv(X)

		return np.array(X, dtype=np.int32) # makes it easier on matplotlib. 

	def log_det(self): 
		logdet = 0.

		for layer in self.layers: 
			if isinstance(layer, tf.keras.layers.InputLayer): 	continue 
			logdet += layer.log_det()
		return logdet


	def loss(self, y_true, y_pred):	 
		#	computes negative log likelihood in bits per dimension. 
		return self.loss_log_det(y_true, y_pred) + self.loss_log_latent_density(y_true, y_pred)

	def loss_log_det(self, y_true, y_pred): 
		# divide by /d to get per dimension and divide by log(2) to get from log base E to log base 2. 
		d			= tf.cast(tf.reduce_prod(y_pred.shape[1:]), 		tf.float32)
		norm		= d * np.log(2.) 
		log_det 	= self.log_det() / norm

		return 		- log_det


	def loss_log_latent_density(self, y_true, y_pred): 
		# divide by /d to get per dimension and divide by log(2) to get from log base E to log base 2. 
		batch_size 	= tf.cast(tf.shape(y_pred)[0], 	tf.float32)
		d			= tf.cast(tf.reduce_prod(y_pred.shape[1:]), 		tf.float32)
		norm		= d * np.log(2.) 
		normal 		= self.latent.log_density(y_pred) / (norm * batch_size)

		return 		- normal

	def compile(self, **kwargs): 
		# overrides what'ever loss the user specifieds; change to complain with exception if they specify it with

		kwargs['loss'] 		= self.loss 

		def lg_det(y_true, y_pred): 	return self.loss_log_det(y_true, y_pred)
		def lg_latent(y_true, y_pred): 	return self.loss_log_latent_density(y_true, y_pred)
		def lg_perfect(y_true, y_pred): return self.loss_log_latent_density(y_true, self.latent.sample(n=1000))

		kwargs['metrics'] = [lg_det, lg_latent, lg_perfect]

		super(Generator, self).compile(**kwargs)

	def fit(self, X, **kwargs): return super(Generator, self).fit(X, y=X, **kwargs)	# if user specifies batch_size here, get upset. 

	def rec(self, X): 

		print(X.shape)
		X, Zs = self.predict(X)#, dequantize=False) # TODO: deactivate dequantize. 
		print(X.shape, [Z.shape for Z in Zs])
		rec = self.predict_inv(X, Zs)
		print(rec.shape)
		return rec

	def check_inv(self, X, precision=10**(-5)): 
		img_shape = X.shape[1:]

		rec = self.rec(X)

		if not np.allclose(X, rec, atol=precision):
			fig, ax = plt.subplots(5, 3)
			for i in range(5): 
				ax[i, 0].imshow(X[i].reshape(img_shape))
				ax[i, 0].set_title("Image")
				ax[i, 1].imshow(rec[i].reshape(img_shape))
				ax[i, 1].set_title("Reconstruction")
				ax[i, 2].imshow((X[i]-rec[i]).reshape(img_shape))
				ax[i, 2].set_title("Difference")
				plt.show()


	def sample(self, n=1000, fix_latent=True):	
		#Z 	= self.latent.sample(n=n, fix_latent=fix_latent)
		
		# Figure out how to handle shape of Z. If no multi-scale arch we want to do reshape below. 
		# If multi-scale arch we don't want to, predict_inv handles it. Figure out who has the responsibility. 

		output_shape 	= self.layers[-1].output_shape[1:]

		X = np.random.normal(0, 1, (n, ) + output_shape).astype(np.float32)

		for layer in self.layers[::-1]: 
			print(layer, X.shape)

			if isinstance(layer, MultiScale): 
				Z = np.random.normal(0, 1, X.shape).astype(np.float32)
				X = layer.call_inv(X, Z)
			else: 
				X = layer.call_inv(X)

		return np.array(X, dtype=np.int32) # makes it easier on matplotlib. 

		return fakes

		
