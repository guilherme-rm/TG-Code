import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
import argparse
from load_data import load_data
import pdb
import torch
from torch_geometric_temporal.nn.recurrent import LRGCN
parser = argparse.ArgumentParser()

parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size to train in one step')
parser.add_argument('--labels', type=int, default=1, help='Number of label classes')
parser.add_argument('--word_pad_length', type=int, default=4438, help='Word pad length for training')
parser.add_argument('--decay_step', type=int, default=500, help='Decay steps')
parser.add_argument('--learn_rate', type=float, default=1e-2, help='Learn rate for training optimization')
parser.add_argument('--train', type=bool, default=False, help='Train mode FLAG')

FLAGS = parser.parse_args()

params = {
        'num_epochs': FLAGS.num_epochs,
        'batch_size':  FLAGS.batch_size,
        'tag_size':  FLAGS.labels,
        'word_pad_length': FLAGS.word_pad_length,
        'feature_dimension':  2,
        'lr':  FLAGS.learn_rate,
        'window_size':  24,
        'num_path':  200,
        'max_path_len': 49
}

n_in = 1
n_out = 1
n_rel = 1
n_bases = 1

model = LRGCN(n_in, n_out, n_rel, n_bases)

train_tuopu_input,train_word_input,test_tuopu_input,test_word_input,ally,ty,whole_mask, path_node_index_array = load_data(params['window_size'])

pdb.set_trace()