import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras import datasets 


##### REAL DATA SET ### 

class mnist(): 

	def __init__(self, digit=-1):  # -1 => all classes
		(self.X, y), (self.X_test, y_test) = datasets.mnist.load_data()
		self.X = self.X.astype(np.float32).reshape(60000, 28, 28, 1)
		#self.X = self.X.reshape(60000, 28**2).astype(np.float32)

		if digit > -1: 
			self.X      = self.X	 [y 	 == digit]
			self.X_test = self.X_test[y_test == digit]
		

	def images(self): 
		return self.X


class cifar10(): 

	def __init__(self, digit=-1):  # -1 => all classes
		(self.X, y), (self.X_test, y_test) = datasets.cifar10.load_data()
		self.X = self.X

	def images(self): 
		return self.X







############### TOY DATASET #################

# Easy
class Normal(): 
	def __init__(self): pass 
	def sample(self, n=1000, mean=0, std=1): return np.random.normal(mean, std, size=(n, 2)).astype(np.float32)

class Uniform(): 
	def __init__(self): pass 
	def sample(self, n=1000, a=0, b=1): 	return np.random.uniform(a, b, size=(n, 2)).astype(np.float32)


# Harder 
"""
	Generate several Gaussians located uniformly on 2d unit circle. 
	See FFJORD page 5 for an example. 
	For simplicity the one below uses 4 Gaussians. 
	TODO: Refactor and generalize to 'n' Gaussians. 

"""
class Gaussians():  
	
	def sample(self, n=100, std=.2): 

		def gaussian(): return np.random.normal(0, std, size=(n, 2))

		# generate gaussians uniformly distributed on circle. 
		# make 'four gaussians' for simplicity. 
		e1 = np.array([0, 1]).reshape(1, 2)
		e2 = np.array([1, 0]).reshape(1, 2)
		
		g1 = gaussian() + e1 
		g2 = gaussian() - e1 
		g3 = gaussian() + e2
		g4 = gaussian() - e2

		return np.concatenate((g1, g2, g3, g4), axis=0).astype(np.float32)

"""
	Generate checkerboard so alternating cells having points. 
	See FFJORD page 5 for an example. 
	For simplicity assumes 4x4 gird. 
	TODO: Refactor and generalize to 'n' cells. 
"""
class Checkboard(): 	 

	def __init__(self): pass 
	def sample(self, n=100): 
		def uniform(): return np.random.uniform(0, 1, size=(n, 2))

		e1 = np.array([0, 1]).reshape(1, 2)
		e2 = np.array([1, 0]).reshape(1, 2)

		u1 = uniform() - 2*e1 
		u2 = uniform() - 2*e2
		u3 = uniform() - 2*e1 - 2*e2
		u4 = uniform() - e1 - e2
		u5 = uniform() + e1 - e2
		u6 = uniform() - e1 + e2
		u7 = uniform() + e1 + e2
		u8 = uniform() 

		return np.concatenate((u1, u2, u3, u4, u5, u6, u7, u8), axis=0).astype(np.float32)


"""
	Generates to "reflected" spirals, see FFJORD page 5 for example. 
	Note: 	
		the reflected spirals are produced using different samples. 
		density is not uniform throughout 'the spiral', i.e., there are more points closer to center. 

"""
class TwoSpirals():  	

	def sample(self): 

		offset = np.array([0, 0.4])
	
		n = np.random.uniform(0, 1, size=(1000, 1)) *4*np.pi# uniform line
		x = -np.cos(n) * n
		y = np.sin(n) * n
		spiral1 =  np.concatenate((x,y), axis=1) + offset

		n = np.random.uniform(0, 1, size=(1000, 1)) *4*np.pi# uniform line
		x = np.cos(n) * n
		y = -np.sin(n) * n
		spiral2 =  np.concatenate((x,y), axis=1) - offset

		return np.concatenate((spiral1, spiral2), axis=0).astype(np.float32)

