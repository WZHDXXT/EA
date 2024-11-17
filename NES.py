import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

DNA_SIZE = 5
POP_SIZE = 3

def get_fitness(pred):
    return pred[:0]**2 + pred[:1]**2

mean = tf.random_normal_initializer([2, ], 13, 1)
print(mean)