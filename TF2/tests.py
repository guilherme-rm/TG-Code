import numpy as np
import tensorflow as tf
import pdb

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    initializer = tf.keras.initializers.he_normal()
    return tf.compat.v1.get_variable(name=name,shape=shape,initializer=initializer)

def glorot_tf2(shape, name=None):
    initializer = tf.keras.initializers.he_normal()
    return tf.Variable(initializer(shape), name=name)

ident = lambda x: x

shape = (2, 2)
name = "weights"
weights = glorot(shape, name)

weights_tf2 = glorot_tf2(shape, name)

pdb.set_trace()