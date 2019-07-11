from invtf 				import Generator
from invtf.visualize 	import visualize_training_2D
from invtf.layers 		import *
from invtf.dataset 		import *
from invtf.dequantize 	import * 
from tensorflow			import keras
from tensorflow.keras 	import Sequential

from tensorflow.keras.layers import ReLU, Dense, Flatten, Reshape, Conv2D
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

			Currently not sure how they have output layer of all coupling networks; do they just have d//2 linear output neurons? 


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

	def mnist(X):   # assumes mnist. 
		n, d = X.shape

		g = Generator(latent.Logistic(d)) 

		# Pre-process steps. 
		g.add(UniformDequantize	(input_shape=[d])) 
		g.add(Normalize			(input_shape=[d]))

		# Build model using additive coupling layers. 
		for i in range(0, 4): 

			ac = AdditiveCoupling(part=i%2, strategy=EvenOddStrategy())
			ac.add(Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="zeros")) 
			ac.add(Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(d//2, bias_initializer="zeros")) 

			g.add(ac) 

		g.add(Affine(exp=True))

		g.compile(optimizer=keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.01, epsilon=10**(-4)))

		g.predict(X[:2])

		return g



class RealNVP(): 

	"""
		1) Change from additive to affine coupling layer: DONE. 
			It seems log computations are fine, but there might be a few issues. 
			!!! The permutation is very simple compared to channel-wise masking / spatial checkerboard.
			TODO: implement channel-wise masking + spatial checkerboard. 

		3) implement squeeze (try different ones)
		
		4) Implement multi-scale architecture. 

		2) Change from dense to residual conv net with ??bnorm???

		Differences: 
			The implementation follows GLOW, so there is not exp in the scaling. 

		Further development: 
		- Do progressive kind of training? 

	"""

	def model(X):  
		input_shape = X.shape[1:]
		d 			= np.prod(input_shape)
		h, w, c 	= input_shape

		g = Generator(latent.Normal(d)) 

		# Pre-process steps. 
		g.add(UniformDequantize	(input_shape=input_shape)) 
		g.add(Normalize			(input_shape=input_shape))

		# Build model using additive coupling layers. 
		g.add(Squeeze())
		h, w, c = h//2, w//2, c*4

		strategy = SplitChannelsStrategy()

		for i in range(0, 2): 
			for j in range(2): 

				ac = AffineCoupling(part=j%2, strategy=strategy)
				ac.add(Flatten())
				ac.add(Dense(100, activation="relu"))
				ac.add(Dense(100, activation="relu"))
				ac.add(Dense(100, activation="relu"))
				ac.add(Dense(d, bias_initializer="ones", kernel_initializer="zeros"))
				ac.add(Reshape((h, w, c)))

				g.add(ac) 

			
			"""g.add(Squeeze())
			h, w, c = h//2, w//2, c*4

			if i == 0 and False: 
				g.add(MultiScale()) # adds directly to output. For simplicity just add half of channels. 
				d = d//2
				c = c//2"""

		g.compile(optimizer=keras.optimizers.Adam(0.0001))

		g.predict(X[:2])

		ac.summary()

		return g


class Glow(): 

	def __init__(self): pass 


class InvResNet(): 

	def __init__(self): pass 



