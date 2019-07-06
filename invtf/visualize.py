import numpy as np
import matplotlib.pyplot as plt 

def visualize_training_2D(g, X): 
	# make code below into a function inside library that 'visualizes' training. 
	fig, ax = plt.subplots(1, 3, figsize=(10, 4))
	plt.ion()

	loss 		= []
	lg_det 		= []
	lg_latent 	= []

	first = True
	while True: 

		for _ in range(10): 		
			history 	= g.fit(X)
			loss		.append(history.history['loss'])
			lg_det		.append(history.history['lg_det'])
			lg_latent	.append(history.history['lg_latent'])

		X		= X
		fakeX 	= g.predict_inv(np.random.normal(0, 1, size=(1000, 2)))

		encX	= g.predict(X)
		realZ	= np.random.normal(0, 1, size=(1000, 2))

		ax[0].cla()
		ax[1].cla()
		ax[2].cla()

		ax[0].plot(fakeX[:, 0], fakeX[:, 1], 'go', alpha=0.3)
		ax[0].plot(X[:, 0], X[:, 1], 'bx', alpha=0.3)
		ax[0].set_title("Data Space X")

		ax[0].set_xlim([-5,5])
		ax[0].set_ylim([-5,5])


		ax[1].plot(realZ[:, 0], realZ[:, 1], 'go', alpha=0.3)
		ax[1].plot(encX[:, 0], encX[:, 1], 'bx', alpha=0.3)
		ax[1].set_title("Latent Space Z")

		ax[1].set_xlim([-5,5])
		ax[1].set_ylim([-5,5])

		ax[2].plot(loss, label="Loss")
		ax[2].plot(lg_det, label="Log Det Jacobian" )
		ax[2].plot(lg_latent, label="Log Latent Density")
		ax[2].set_xlabel("Epochs")
		ax[2].set_ylabel("Negative Log-Likelihood")
		ax[2].legend()

		if first: 
			plt.tight_layout()
			first = False

		plt.pause(.1)

