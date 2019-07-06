from invtf 				import Generator
from invtf.visualize 	import visualize_training_2D
from invtf.layers 		import Linear, Affine, CoupledReLU
from invtf.dataset 		import Moons, Normal
from tensorflow			import keras

"""
	Learn a Gaussian N(3, 0.5) using a single Affine layer. 
"""

X = Normal().sample(n=1000, mean=3, std=1) 

# Translate from one normal to another, very simple. 
g = Generator()
g.add(Affine())

g.compile(optimizer=keras.optimizers.Adam(0.001))
g.build(input_shape=X.shape)
g.summary()

g.check_inv(X) 

visualize_training_2D(g, X)

