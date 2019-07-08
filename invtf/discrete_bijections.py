


import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np



class Exponential(): pass


class Simple(): 

	def call(self, X): 

		# assumes 'flat' 
		
		even = X[::2]
		odd  = X[1::2]

		output = ((even + odd) * (even + odd + 1)) // 2 + odd

		return output 



	def call_inv(self, Z): 
		# assumes flat 


	def log_det(self): return .0
