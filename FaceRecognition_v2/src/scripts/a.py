import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Defina o índice da sua GPU aqui

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
