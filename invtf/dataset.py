import numpy as np

class Normal(): 
	def __init__(self): pass 
	def sample(self, n=100, mean=0, std=1): return np.random.normal(mean, std, size=(n, 2))

class Moons(): 
	def __init__(self): pass
	def sample(self, n=100): pass


class SixGaussians(): 
	def __init__(self): pass
	def sample(self, n=100): pass 


class Text(): 
	def __init__(self): pass 
	def sample(self, n="Hello World"): pass

