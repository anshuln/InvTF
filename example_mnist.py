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

from invtf.models import NICE

import invtf.latent as latent


X = mnist(digit=-1).images()
g = NICE.mnist(X)

g.summary()

#g.check_inv(X[:100], precision=10**(-0))  # I think uniform noise gets in here!! 

fig, ax = plt.subplots()


log_dir			= "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard 	= keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

g.fit(X, batch_size=256, callbacks=[tensorboard], epochs=100)

fake = g.sample(1, fix_latent=True)

ax.imshow(fake.reshape(28, 28), vmin=0, vmax=2**8) 
plt.pause(.1)

