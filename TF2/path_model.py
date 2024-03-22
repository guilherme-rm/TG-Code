from layers import *
import tensorflow as tf
import pdb


class LRGCN(object):
  def lstm_cell(self,previous_h_c_tuple,input,tuopu,tuopu_pre):
    state,cell = tf.unstack(previous_h_c_tuple)
    feature = input 
    i = tf.nn.sigmoid(self.GCN_layer1_i(inputs = state, tuopu = tuopu_pre) + self.GCN_layer2_2_i(inputs = self.GCN_layer2_1_i(inputs = feature, tuopu = tuopu), tuopu = tuopu))
    f = tf.nn.sigmoid(self.GCN_layer1_f(inputs = state, tuopu = tuopu_pre) + self.GCN_layer2_2_f(inputs = self.GCN_layer2_1_f(inputs = feature, tuopu = tuopu), tuopu = tuopu))
    o = tf.nn.sigmoid(self.GCN_layer1_o(inputs = state, tuopu = tuopu_pre) + self.GCN_layer2_2_o(inputs = self.GCN_layer2_1_o(inputs = feature, tuopu = tuopu), tuopu = tuopu))
    cell = tf.multiply(f,cell) + tf.multiply(i,tf.nn.tanh(self.GCN_layer1_c(inputs = state, tuopu = tuopu_pre) + self.GCN_layer2_2_c(inputs = self.GCN_layer2_1_c(inputs = feature, tuopu = tuopu), tuopu = tuopu)))
    state = tf.multiply(o,tf.nn.tanh(cell))
    return tf.stack([state,cell])

  def lstm_rgcn(self,inputs,tuopu,initial_state,rgcn_input_dim,rgcn_output_dim,hidden_dim = 96, zero_start = True):
      if zero_start:
        initializer = tf.keras.initializers.Zeros()
        H = tf.Variable(initializer(shape=(self.n, self.u)), trainable=False, name='initial_state')
        C = tf.Variable(initializer(shape=(self.n, self.u)), trainable=False, name='initial_cell')
      else:
        H = initial_state 
        C = initial_state 

      previous_h_c_tuple = tf.stack([H,C])
      self.GCN_layer1_i = GraphConvolution(input_dim=rgcn_output_dim, output_dim=rgcn_output_dim, placeholders=self.placeholders,
                                           act=tf.identity,dropout=True,logging=True)
      self.GCN_layer1_f = GraphConvolution(input_dim=rgcn_output_dim, output_dim=rgcn_output_dim, placeholders=self.placeholders,
                                           act=tf.identity,dropout=True,logging=True)
      self.GCN_layer1_o = GraphConvolution(input_dim=rgcn_output_dim, output_dim=rgcn_output_dim, placeholders=self.placeholders,
                                           act=tf.identity,dropout=True,logging=True)
      self.GCN_layer1_c = GraphConvolution(input_dim=rgcn_output_dim, output_dim=rgcn_output_dim, placeholders=self.placeholders,
                                           act=tf.identity,dropout=True,logging=True)
      self.GCN_layer2_1_i = GraphConvolution(input_dim=rgcn_input_dim, output_dim=hidden_dim, placeholders=self.placeholders,
                                             act=tf.identity,dropout=True,sparse_inputs=False,logging=True)
      self.GCN_layer2_1_f = GraphConvolution(input_dim=rgcn_input_dim, output_dim=hidden_dim, placeholders=self.placeholders,
                                             act=tf.identity,dropout=True,sparse_inputs=False,logging=True)
      self.GCN_layer2_1_o = GraphConvolution(input_dim=rgcn_input_dim, output_dim=hidden_dim, placeholders=self.placeholders,
                                             act=tf.identity,dropout=True,sparse_inputs=False,logging=True)
      self.GCN_layer2_1_c = GraphConvolution(input_dim=rgcn_input_dim, output_dim=hidden_dim, placeholders=self.placeholders,
                                             act=tf.identity,dropout=True,sparse_inputs=False,logging=True)
      self.GCN_layer2_2_i = GraphConvolution(input_dim=hidden_dim, output_dim=rgcn_output_dim, placeholders=self.placeholders
                                             ,act=tf.identity,dropout=True,logging=False)
      self.GCN_layer2_2_f = GraphConvolution(input_dim=hidden_dim, output_dim=rgcn_output_dim, placeholders=self.placeholders,
                                             act=tf.identity,dropout=True,logging=False)
      self.GCN_layer2_2_o = GraphConvolution(input_dim=hidden_dim, output_dim=rgcn_output_dim, placeholders=self.placeholders,
                                             act=tf.identity,dropout=True,logging=False)
      self.GCN_layer2_2_c = GraphConvolution(input_dim=hidden_dim, output_dim=rgcn_output_dim, placeholders=self.placeholders,
                                             act=tf.identity,dropout=True,logging=False)
      inputs = inputs
      outputs = []
      
      for i in range(self.window_size):
        
        adj_time = tuopu[i]
        pre_adj_time = tuopu[i] if i==0 else tuopu[i-1]

        input_F = tf.gather(inputs,[i]) 
        input_F = tf.reshape(input_F,(self.n,-1))
        previous_h_c_tuple = self.lstm_cell(previous_h_c_tuple,input_F,adj_time,pre_adj_time)
        outputs.append(tf.unstack(previous_h_c_tuple)[0])
      
      return outputs,outputs[-1]

  def elapsed_cell(self,state,input,GCN1,GCN2_1,GCN2_2):
    h1 = GCN1(state)
    hidden = GCN2_1(input)
    h2 = GCN2_2(hidden)
    state = h1 + h2
    return state

  def elapsed_rgcn(self,inputs,initial_state,rgcn_input_dim,rgcn_output_dim,hidden_dim = 24,zero_start = True):
    if zero_start:
      initializer = tf.keras.initializers.Zeros(shape=(self.n, self.u))
      H = tf.Variable(initializer, trainable=False, name='initial_state')
    else:
      H = initial_state 
    GCN_layer1 = GraphConvolution(input_dim=rgcn_output_dim,output_dim=rgcn_output_dim,placeholders=self.placeholders,act=tf.identity,dropout=True,logging=True)
    GCN_layer2_1 = GraphConvolution(input_dim=rgcn_input_dim,output_dim=hidden_dim,placeholders=self.placeholders,act=tf.nn.relu,dropout=True,sparse_inputs=False,logging=True)
    GCN_layer2_2 = GraphConvolution(input_dim=hidden_dim,output_dim=rgcn_output_dim,placeholders=self.placeholders,act=tf.identity,dropout=True,logging=False)
    outputs = []
    inputs = inputs
    for i in range(self.window_size):
      input_F = tf.gather(inputs,[i]) 
      input_F = tf.reshape(input_F,(626,-1))
      H = self.elapsed_cell(H,input_F,GCN_layer1,GCN_layer2_1,GCN_layer2_2)
      outputs.append(H)

    return outputs,H
      
  def build_graph(self, placeholders, n=626, d=2, u=8, d_a=32, r=10, window_size=24,reuse=False):
   
    self.n = n
    self.d = d
    self.d_a = d_a
    self.u = u
    self.r = r
    self.window_size = window_size

    initializer = tf.keras.initializers.he_normal()

    self.placeholders = placeholders
    self.input_F = placeholders['features']
    self.tuopu = placeholders['support']
    _,self.H = self.lstm_rgcn(initial_state = None, inputs = self.input_F, tuopu = self.tuopu, rgcn_input_dim = self.d, rgcn_output_dim = self.u)
    self.W_s1 = tf.Variable(initializer(shape=(self.d_a, self.u)), name='W_s1')
    self.W_s2 = tf.Variable(initializer(shape=(self.r, self.d_a)), name='W_s2')
    self.batch_size = tf.shape(self.input_F)[0]
    self.A = A = tf.nn.softmax(tf.matmul(self.W_s2, tf.tanh(tf.matmul(self.W_s1, tf.transpose(self.H)))))
    self.M = tf.matmul(A, self.H)
    out_num = 8
    self.out_Mb = tf.Variable(initializer(shape=(self.n, self.r)), name='out_Mb')
    initial_s = tf.matmul(self.out_Mb,self.M)      
    self.outputs,self.H2 = self.lstm_rgcn(initial_state = initial_s,inputs = self.input_F,tuopu = self.tuopu, rgcn_input_dim = self.d,rgcn_output_dim = out_num,zero_start = False)