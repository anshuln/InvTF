from invtf 				import Generator
from invtf.visualize 	import visualize_training_2D
from invtf.layers 		import *
from invtf.dataset 		import *
from invtf.dequantize 	import * 
from tensorflow			import keras
from tensorflow.keras 	import Sequential
import datetime

from tensorflow.keras.layers import ReLU, Dense
from tensorflow.keras.models import Sequential

from invtf.models import *

import invtf.latent as latent


fig, ax = plt.subplots(1, 3)
for i in range(3): ax[i].axis('off')

#X = mnist().images()
#img_shape = X.shape[1:-1]

X = cifar10().images()
img_shape = X.shape[1:]

g = RealNVP.model(X)

g.summary()

g.check_inv(X[:2])


for i in range(300): 
	g.fit(X, batch_size=512, epochs=1)

	fake = g.sample(1, fix_latent=True)

	ax[0].imshow(fake.reshape(img_shape))
	ax[0].set_title("Fake")

	ax[1].imshow(X[0].reshape(img_shape))
	ax[1].set_title("Real")

	ax[2].imshow(g.rec(X[:1]).reshape(img_shape))
	ax[2].set_title("Reconstruction")
	if i == 0: plt.tight_layout()

	plt.pause(.1)
