"""
	I experienced issues with memory allocation when using Conv2D. 
	For example: Training simple CNN from tensorflow 2.0 beginner guide throwed a "cuda conv algo error". 
	I tried installing >5 different combinations of NVIDIA Driver, Cuda and Cudnn. 
	None of the attempted combinations fixed the issue.

	Changing the code to dynamically allocate memory fixed the issue.
	This code can easily be removed by commenting out "import invtf.grow_memory" in "__init__.py". 
"""
import tensorflow as tf 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
