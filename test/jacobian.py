
import unittest
import sys
sys.path.append("../")
import invtf
import invtf.layers
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import warnings


class TestJacobian(unittest.TestCase): 

	def assertJacobian(self, g, X): 
		"""
			Input: 		
				g:		Model which has call(X) return a tensor that depends on X. 
				X:		Test data. 
	
			Computes the jacobian of g(X) wrt X using tf.GradientTape and compares 
			the log determinant with that computed using log_det. 

			The stability of determinant computations are not that good, so the 
			np.allclose has a quite high absolute tolerance. 

		"""
		with tf.GradientTape() as t: 
			t.watch(X)
			z = g.call(X)

		J = t.jacobian(z, X)
		J = tf.reshape(J, (32*32*3, 32*32*3))

		lgdet1 = tf.math.log(tf.linalg.det(J)).numpy()
		lgdet2 = g.log_det().numpy()

		print(lgdet1, lgdet2)
		
		# If the following equation is element-wise True, then allclose returns True.
		# 		absolute(a - b) <= (atol + rtol * absolute(b))
		# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
		A = np.allclose( lgdet1, lgdet2 , atol=10**(-4), rtol=0.1)  # deciding these values are very difficult for all tests.. 

		self.assertTrue(A)
		#print("\t", lgdet1, lgdet2, "\t", end="")
		

	def test_actnorm_init(self): 
		X = tf.random.normal((1, 32, 32, 3), 0, 1)
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.ActNorm()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertJacobian(g, X)

	def test_actnorm_fit(self): 
		X = tf.random.normal((1, 32, 32, 3), 0, 1)
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.ActNorm()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		g.fit(X[:1], verbose=False)
		self.assertJacobian(g, X)


	def test_invconv_init(self): 
		X = tf.random.normal((1, 32, 32, 3), 0, 1)
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Inv1x1Conv()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertJacobian(g, X)

	def test_invconv_fit(self): 
		X = tf.random.normal((1, 32, 32, 3), 0, 1)
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Inv1x1Conv()) 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		g.fit(X[:1], verbose=False)
		self.assertJacobian(g, X)

	def test_glow_init(self): 
		X = tf.random.normal((1,32,32,3), 0, 1)
		g = invtf.models.Glow.model(X)
		self.assertJacobian(g, X)

	def test_glow_fit(self): 
		X = tf.random.normal((1, 32,32,3), 0, 1)
		g = invtf.models.Glow.model(X)
		g.fit(X[:1], verbose=False)
		self.assertJacobian(g, X)

	def test_3dconv_init(self): 
		X = tf.random.normal((1, 32,32,3), 0, 1)
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Conv3DCirc()) # initialize not like ones so it becomes zero. 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.predict(X[:1])
		self.assertJacobian(g, X)

	def test_3dconv_fit(self): 
		X = tf.random.normal((1, 32,32,3), 0, 1)
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Conv3DCirc()) # initialize not like ones so it becomes zero. 
		g.compile(optimizer=keras.optimizers.Adam(0.001))
		g.fit(X[:1]) 
		self.assertJacobian(g, X)



