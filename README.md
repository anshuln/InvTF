# InvTF
Train invertible generative models using the simplicity of Keras.

<b>Example:</b> Invertible Residual Networks.  

```
from invtf import Flow, InvResNet, faces

flow = Flow()

for _ in range(10): 
    flow.add(InvResNet())

flow.compile()
flow.fit(faces())
```

<img src="faces.png">

Most recent reversible generative model [1,2,3,4] have been <a href="">reproduced</a> in InvTF. Pretrained models are automatically downloaded when needed.

<b>Example</b>: Use pretrained model.

```
from invtf import Glow

glow.interpolate(faces()[0], faces()[1])

glow.generate(10)
```

<img src="interpolate.png">
<img src="generated.png">

Please see our <a href="">tutorial</a> and <a href="">documentation</a> for more information. 

TLDR: Easily train reversible generative models with 
