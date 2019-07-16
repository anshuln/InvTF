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


class TestInverse(unittest.TestCase): 

	X = keras.datasets.cifar10.load_data()[0][0][:1] # a single cifar image. 

	def assertInverse(self, g, X): 
		"""
			Input: 		
				g:		Model which has call(X) return a tensor that depends on X. 
				X:		Test data. 
	
			Computes the reconstruction of X and compares with X. 

		""" 
		rec = g.rec(X)
		A = np.allclose(rec, X, atol=1, rtol=0.1) # assumes data is in bytes. 

		# find entry with largest difference and print their relative values. 
		diff = np.abs(A - rec)
		entry = np.argmax(diff)

		if A is False: 
			fig, ax = plt.subplots(3, 1)
			ax[0].imshow(rec.reshape(32,32,3)/255)
			ax[1].imshow(X.reshape(32,32,3)/255)
			ax[2].imshow((rec-X).reshape(32,32,3)/255)
			plt.show()

		self.assertTrue(A)
		
		print("\t", X.reshape(X.size)[entry], rec.reshape(rec.size)[entry], end="")

	def test_actnorm_init(self): 
		X = TestInverse.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.ActNorm()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertInverse(g, X)

	def test_actnorm_fit(self):
		X = TestInverse.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.ActNorm()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		g.fit(X[:1], verbose=False) 
		self.assertInverse(g, X)


	def test_3dconv_init(self): 
		X = TestInverse.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Conv3DCirc()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertInverse(g, X)

	def test_3dconv_fit(self): 
		X = TestInverse.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Conv3DCirc()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.fit(X[:1], epochs=1, verbose=False) 
		self.assertInverse(g, X)


	def test_invconv_init(self):  
		X = TestInverse.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Inv1x1Conv()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertInverse(g, X)

	def test_invconv_fit(self):  
		X = TestInverse.X 
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Inv1x1Conv()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		g.fit(X[:1], epochs=1, verbose=False)
		self.assertInverse(g, X)
		


