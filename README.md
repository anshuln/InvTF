*cite us using <a href="">bibtex</a>*
# InvTF
Train invertible generative models using the simplicity of Keras (aka normalizing flows).

<b>Example:</b> Invertible Residual Networks.  

```
from invtf import Generator, InvResNet, faces

gen = Generator()

for _ in range(10): 
    gen.add(InvResNet())

gen.compile()
gen.fit(faces())
```

<img src="faces.png">

Most recent invertible generative model [1,2,3,4] have been <a href="">reproduced</a> in InvTF. Pretrained models are automatically downloaded when needed.

<b>Example</b>: Use pretrained model.

```
from invtf import Glow

glow.interpolate(faces()[0], faces()[1])

glow.generate(10)
```

<img src="interpolate.png">
<img src="generated.png">

Please see our <a href="">tutorial</a> and <a href="">documentation</a> for more information. 

TLDR: Easily train reversible generative models with Keras.

# Details

"invtf/"	the package directory. 

"articles/":	 notes / ipython notebook explaining previous articles reproducing their experiments in ipython notebooks.

"examples": 	implementation of a lot of examples, naming conforms with "dataset" x "model".py
		examples:
		"syntheticNormal_feedforward.py"
		"cifar10_glow.py"
		Besides show-casing how the framework can be used, these files are also used for test casing, just run 'run_all.py' (might take itme). 


# Development details

Developed using tensorflow-gpu 2.0beta, ubuntu 16.04 LTS, cuda 10, RTX 2080 8 GB


