import numpy as np
import tensorflow as tf
import pdb
from scipy import sparse as sp


ident = lambda x: x
window_size  = 1

tf.compat.v1.disable_eager_execution()

word_pad_length = 4438
num_path = 200
max_path_len = 49
feature_dimension = 2

placeholders = {
    'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(window_size)],
    'features':tf.compat.v1.placeholder(tf.float32, shape=(window_size,word_pad_length,feature_dimension)),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(num_path,)),
    'labels_mask': tf.compat.v1.placeholder(tf.int32,shape=(num_path,word_pad_length)),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'path_node_index_array': tf.compat.v1.placeholder(tf.int32,shape=(num_path, max_path_len)),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}

pdb.set_trace()

"""

def main(FLAGS):
    print("Learning Rate:", FLAGS.learning_rate)
    print("Batch Size:", FLAGS.batch_size)
    print("Epochs:", FLAGS.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)

"""