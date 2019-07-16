import unittest
import sys
sys.path.append("../")
import invtf.latent
# import invtf.layers
#from tensorflow.python.ops.parallel_for.gradients import jacobian
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from invtf.generator_const_backprop import Generator as GenConst
from invtf.layers_const_backprop import *
from tensorflow.keras.layers import ReLU, Dense, Flatten, Conv2D	

class GeneratorGradTest(GenConst):
	def compute_gradients(self,X):
		x = self.call(X)        #I think putting this in context records all operations onto the tape, thereby destroying purpose of checkpointing...
		last_layer = self.layers[-1]
		d = np.prod(X.shape[1:])
		#Computing gradients of loss function wrt the last acticvation
		with tf.GradientTape() as tape:
			tape.watch(x)
			loss = self.loss(x)    #May have to change
		grads_combined = tape.gradient(loss,[x])
		dy = grads_combined[0]
		y = x
		#Computing gradients for each layer
		gradients = []
		for layer in self.layers[::-1]:     
			x = layer.call_inv(y)
			dy,grads = layer.compute_gradients(x,dy,layer.log_det,d*np.log(2.))	#TODO implement scaling here -- DONE
			gradients+=(grads)
			y = x 
		if len(gradients) > 1:	#TODO better fix for issue with only single gradient
			return [gradients[-2]]+[gradients[-1]]+gradients[0:-2][::-1]
		else:
			return gradients

	def actual_gradients(self,X):
		with tf.GradientTape() as tape:
			loss = self.loss(self.call(X))
		grads = tape.gradient(loss,self.trainable_variables)
		return grads

class TestGradients(unittest.TestCase):
	X = keras.datasets.cifar10.load_data()[0][0][:5].astype('f') # a single cifar image batch.

	def assertGrad(self,g,X):
		computed_grads = g.compute_gradients(X)
		actual_grads = g.actual_gradients(X) 
		A = [np.allclose(np.abs(x[0]-x[1]),0,atol=1, rtol=0.1) for x in zip(computed_grads,actual_grads) if x[0] is not None]
		# print("computed",computed_grads,"actual_grads",actual_grads)
		print("Max discrepancy in gradients",np.max(np.array([(np.max(np.abs(x[0]-x[1]))) for x in zip(computed_grads,actual_grads) if x[0] is not None])))
		self.assertTrue(np.array(A).all())

	def test_circ_conv(self):
		X = TestGradients.X 
		d = 32*32*3
		g = GeneratorGradTest(invtf.latent.Normal(d)) 
		g.add(Conv3DCirc())
		g.predict(X[:1])
		self.assertGrad(g,X)		

	def test_inv_conv(self):
		X = TestGradients.X 
		d = 32*32*3
		g = GeneratorGradTest(invtf.latent.Normal(d)) 
		g.add(Inv1x1ConvPLU())
		g.predict(X[:1])
		self.assertGrad(g,X)		

	def test_act_norm(self):
		X = TestGradients.X 
		d = 32*32*3
		g = GeneratorGradTest(invtf.latent.Normal(d)) 
		g.add(ActNorm())
		g.predict(X[:1])
		self.assertGrad(g,X)		

	def test_affine_coupling(self):
		X = TestGradients.X 
		d = 32*32*3
		g = GeneratorGradTest(invtf.latent.Normal(d)) 
		b = AffineCoupling()
		b.add(Flatten())
		b.add(Dense(d,activation='relu'))
		g.add(Squeeze())
		g.add(b)
		# g.predict(X[:1])
		self.assertGrad(g,X)		

