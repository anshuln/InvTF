import unittest
import sys
sys.path.append("../")
import invtf
import invtf.layers
#from tensorflow.python.ops.parallel_for.gradients import jacobian
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt 
import warnings
from tensorflow.keras.layers import *


class TestIdentityInit(unittest.TestCase): 

	X = keras.datasets.cifar10.load_data()[0][0][:1].astype(np.float32) # a single cifar image. 

	def assertIdentityInit(self, g, X): 
		"""
			Input: 		
				g:		Model which has call(X) return a tensor that depends on X. 
				X:		Test data. 
	
			Computes the reconstruction of X and compares with X. 

		""" 
		enc = g.predict(X)[0].numpy()

		if enc.shape != X.shape: enc = enc.reshape(X.shape) # this is a hack and doesn't work generally. 
		
		A = np.allclose(enc, X, atol=1, rtol=0.1) # assumes data is in bytes. 

		if A is False: 
			fig, ax = plt.subplots(3, 1)
			ax[0].imshow(enc.reshape(32,32,3)/255)
			ax[1].imshow(X.reshape(32,32,3)/255)
			ax[2].imshow((enc-X).reshape(32,32,3)/255)
			plt.show()

		self.assertTrue(A)

	def test_actnorm_init(self): 
		X = TestIdentityInit.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.ActNorm()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertIdentityInit(g, X)


	def test_3dconv_init(self): 
		X = TestIdentityInit.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Conv3DCirc()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertIdentityInit(g, X)

	def test_affine_coupling_init(self): 
		X = TestIdentityInit.X
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 

		width = 32
		c = 12

		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.layers.Squeeze())

		ac = invtf.layers.AffineCoupling(part=0, strategy=invtf.coupling_strategy.SplitChannelsStrategy())
		ac.add(Conv2D(width, kernel_size=(3,3), activation="relu", padding="SAME", 
						kernel_initializer="normal", bias_initializer="zeros")) 
		ac.add(Conv2D(width, kernel_size=(1,1), activation="relu", padding="SAME", 
						kernel_initializer="normal", bias_initializer="zeros"))
		ac.add(Conv2D(c, kernel_size=(3,3), 				   padding="SAME",
						kernel_initializer="zeros", bias_initializer="ones"))  # they add 2 here and apply sigmoid. 

		g.add(ac)

		g.add(invtf.layers.UnSqueeze())

		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertIdentityInit(g, X)


	def test_glow_init(self):  # without inv 1x1

		X = TestIdentityInit.X
		g = invtf.models.Glow.model(X)

		print(g.log_det())
		pred = g.predict(X)[0]
		print(g.loss(pred, pred))

		self.assertIdentityInit(g, X)



	"""def test_invconv_init(self):  
		X = TestIdentityInit.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Inv1x1Conv()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertIdentityInit(g, X)"""

