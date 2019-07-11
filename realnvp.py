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

from invtf.models import NICE, RealNVP

import invtf.latent as latent


fig, ax = plt.subplots(3, 1)

X = mnist().images()
img_shape = X.shape[1:-1]

#X = cifar10().images()
#img_shape = X.shape[1:]

g = RealNVP.model(X)

g.summary()

g.check_inv(X[:2])


log_dir			= "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard 	= keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

for _ in range(300): 
	g.fit(X, batch_size=512, callbacks=[tensorboard], epochs=1)

	fake = g.sample(1, fix_latent=True)

	ax[0].imshow(fake.reshape(img_shape))
	ax[0].set_title("Fake")

	ax[1].imshow(X[0].reshape(img_shape))
	ax[1].set_title("Real")

	ax[2].imshow(g.rec(X[:1]).reshape(img_shape))
	ax[2].set_title("Reconstruction")

	plt.pause(.1)

	g.check_inv(X[:1])
