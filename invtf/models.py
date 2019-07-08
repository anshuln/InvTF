

class NICE(): 
	"""
		Returns a model ready to train. A few issues here and there, but even with the bugs it produces 
		0's if trained on MNSIT with a single digit after very few iterations. 
	"""

	def model(X): 
		n, d = X.shape

		g = Generator() 

		# Pre-process steps. 
		g.add(UniformDequantize	(input_shape=[d])) 
		g.add(Normalize			(input_shape=[d]))

		# Build model using additive coupling layers. 
		for i in range(0,4): 

			ac = AdditiveCoupling(part=i%2, strategy=EvenOddStrategy())
			ac.add(Dense(d//2, activation="relu")) 
			ac.add(Dense(d//2, activation="relu"))
			ac.add(Dense(d//2, activation="relu"))
			ac.add(Dense(d//2, activation="relu"))
			ac.add(Dense(d//2))

			g.add(ac) 

		g.add(Affine())

		g.compile(optimizer=keras.optimizers.Adam(0.001))

		g.predict(X[:2])

		return g 





class RealNVP(): 

	def __init__(self): pass 


class Glow(): 
	def __init__(self): pass 


class InvResNet(): 

	def __init__(self): pass 



