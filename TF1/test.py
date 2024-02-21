import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Run on the GPU
c = tf.matmul(a, b)
print(c)
