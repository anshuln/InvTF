from invtf.dataset 	import *
from invtf.models import *
import argparse
import os 

"""
	DISCLAIMER: 

		This is just a placeholder file, the current architectures are substantially smaller than that
		described in their respective articles. Furthermore, some functionality has not been implemented
		yet, the results are thus substantially worse than that reported in the original articles. 

"""

if __name__ == "__main__": 

	parser = argparse.ArgumentParser()

	parser.add_argument("--problem", type=str, default='mnist', help="Problem (mnist/cifar10")  # add fmnist, cifar100, celeb, svnh, rooms. 
	parser.add_argument("--model", 	 type=str, default='nice',  help="Model (nice/realnvp/glow/flow++/iresnet")  # add fmnist, cifar100, celeb, svnh, rooms. 
		
	args = parser.parse_args()


	# Load Dataset (refactor into just giving args.problem and we get data/imgshape) 
	if args.problem == "mnist": 
		X = mnist().images()
		img_shape = X.shape[1:-1]
	if args.problem in ["fmnist", "fashion_mnist","fashionmnist", "fasion-mnist"]: 
		X = fmnist().images()
		img_shape = X.shape[1:-1]
	if args.problem in ["cifar10", "cifar"]:  
		X = cifar10().images()
		img_shape = X.shape[1:]
	if args.problem == "cifar100":  
		X = cifar100().images()
		img_shape = X.shape[1:]

	# TO BE IMPLEMENTED
	if args.problem in ["celeba", "celeb", "faces"]:  
		X = celeba.load(5000).astype(np.float32)[:, ::8, ::8, :]
		print(X.shape)
		img_shape = X.shape[1:]
	if args.problem == "imagenet32":  raise NotImplementedError()

	# Get Model. 
	if args.model == "nice":  	
		X = X.reshape((X.shape[0], -1))
		g = invtf.models.NICE.model(X)
	if args.model == "realnvp": g = invtf.models.RealNVP.model(X)
	if args.model == "glow":    g = invtf.models.Glow.model(X)
	if args.model == "flowpp":  g = invtf.models.FlowPP.model(X)
	if args.model == "conv3d":  g = invtf.models.Conv3D.model(X)
	if args.model == "iresnet": raise NotImplementedError()

	# Print summary of model. 
	g.summary()

	# Initialize plots (TODO: window position might break from Ubuntu -> win/mac)
	fig, ax = plt.subplots(1, 3)
	for i in range(3): ax[i].axis("off")
	fig.canvas.manager.window.wm_geometry("+0+0")
	fig_loss, ax_loss = plt.subplots()
	fig.canvas.manager.window.wm_geometry("+800+0")

	# Initialize folder to save training history plots. 
	histories = {}
	folder_path = "reproduce/" + args.problem + "_" + args.model + "/"
	if not os.path.exists(folder_path): os.makedirs(folder_path)

	# Train model for epochs iterations. 
	epochs = 1000000
	for i in range(epochs): 

		history = g.fit(X, batch_size=128, epochs=1)	

		# Init histories to fit the history object. 
		if histories == {}: 
			for key in history.history.keys(): histories[key] = []
				
		# Update loss plot. 
		ax_loss.cla()
		for j, key in enumerate(history.history.keys()): 
			histories[key] += history.history[key]
			ax_loss.plot(np.arange(len(histories[key])), histories[key], "C%i"%j, label=key)

		ax_loss.legend()
		ax_loss.set_ylabel("NLL")
		ax_loss.set_xlabel("Epochs")
		ax_loss.set_ylim([0, 10])
		ax_loss.set_xlim([0, epochs])
		ax_loss.set_title(args.problem + " " + args.model)
		
		# Plot fake/real/reconstructed image. 
		fake = g.sample(1, fix_latent=True)

		ax[0].imshow(fake.reshape(img_shape)/255)
		ax[0].set_title("Fake")

		ax[1].imshow(X[0].reshape(img_shape)/255)
		ax[1].set_title("Real")

		ax[2].imshow(g.rec(X[:1]).reshape(img_shape)/255)
		ax[2].set_title("Reconstruction")
		if i == 0: plt.tight_layout()

		plt.pause(.1)

		fig.savefig(folder_path + "img_%i.png"%(i+1))
		fig_loss.savefig(folder_path + "loss.png")

