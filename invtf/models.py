from invtf 				import Generator
from invtf.visualize 	import visualize_training_2D
from invtf.layers 		import *
from invtf.dataset 		import *
from invtf.dequantize 	import * 
from tensorflow			import keras
from tensorflow.keras 	import Sequential

from tensorflow.keras.layers import ReLU, Dense
from tensorflow.keras.models import Sequential

import invtf.latent as latent


class NICE(): 

	"""
		Learn MNIST using (something close to) NICE architecture. 
		They get around 4.36

		Experiments: 
			- compare normal vs logistic distribution. 
			- try zero initialization. 
			- compare with conv networks instead of dense. 
			- they normalize to [0, 1], try [-1, +1] instead. 
			  Furthermore, try identity initialization and chose interval to "fit" gaussian. 

			(note: they train for 1500 epochs and cross validate). 

		Comments:
			Issue 1: 
			In [1] they write "... we add a uniform noise of 1/256 to the data and rescale it to be in [0,1] after dequantization. "
			It is not clear if they normalize to [0,1] first and then add 1/256, or they add 1/256 to the {0,1,2,...,2**8-1}. 
			I added U(0,1) to {0,1,2,...,2**8-1} as outlined in the flow++ article. 


		Lacking information:
			- No batch size is reported. 
			- No information on weight initialization. 
			- No pseudo-code. 


		Differences:
			- Potentially the dequantization (see comments above) 
			- The scaling in the end has bias which they do not have. 

		Current issues: 
			- Training yields NAN after a few epochs, besides that everything seems to be working.  Get tensorbaord to work and start debugging. It seems this usually happens around when the following two gets close: 
					lg_latent: 2.7125 - lg_perfect: 2.8825 
		

	"""

	def mnist(X):  
		n, d = X.shape

		g = Generator(latent.Logistic(d)) 

		# Pre-process steps. 
		g.add(UniformDequantize	(input_shape=[d])) 
		g.add(Normalize			(input_shape=[d]))

		# Build model using additive coupling layers. 
		for i in range(0, 4): 

			ac = AdditiveCoupling(part=i%2, strategy=EvenOddStrategy())
			ac.add(Dense(d//2, activation="relu", bias_initializer="zeros", kernel_initializer="zeros")) 
			ac.add(Dense(d//2, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(d//2, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(d//2, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(d//2, bias_initializer="zeros")) 

			g.add(ac) 

		g.add(Affine(exp=True))

		g.compile(optimizer=keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.01, epsilon=10**(-4)))

		g.predict(X[:2])

		return g



class RealNVP(): 

	def __init__(self): pass 


class Glow(): 
	def __init__(self): pass 


class InvResNet(): 

	def __init__(self): pass 



