# import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D
# import utils
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
tf.config.set_visible_devices([], 'GPU')

LAYER_1_FILTER_N = 30
LAYER_2_NEURON_N = 30

def threshold(x):
  return tf.nn.relu(tf.math.sign(x))

class CriticalNCA(tf.keras.Model):
  def __init__(self, kernel_init="zeros", hidden_channel_n=4,
               neighborhood=3,
               layer_1_filter_n=LAYER_1_FILTER_N,
               layer_2_neuron_n=LAYER_2_NEURON_N):
    super().__init__()
    self.hidden_channel_n = hidden_channel_n
    self.channel_n = self.hidden_channel_n + 1
    self.neighborhood = neighborhood
    self.padding_size = self.neighborhood // 2

    self.layer_1_filter_n = layer_1_filter_n
    self.layer_2_neuron_n = layer_2_neuron_n

    if kernel_init == "zeros":
      kernel_initializer = tf.zeros_initializer
    elif kernel_init == "random":
      kernel_initializer = tf.initializers.GlorotUniform

    self.dmodel = tf.keras.Sequential([
          Conv1D(self.layer_1_filter_n, self.neighborhood,
                 activation=tf.nn.relu, padding="VALID"),
          Conv1D(self.layer_2_neuron_n, 1, activation=tf.nn.relu),
          Conv1D(self.channel_n, 1, activation=threshold,
                 kernel_initializer=kernel_initializer)
    ])

    self(tf.zeros([1, self.neighborhood, self.channel_n]))  # dummy calls to build the model


  def call(self, x):
    xx = tf.tile(x, [1,3,1])
    ca_size = tf.shape(x)[1]

    x_padded = xx[:,ca_size-self.padding_size:2*ca_size+self.padding_size]

    return self.dmodel(x_padded)

  def get_dict_args(self):
    nca_dict = dict(self.__dict__)
    del nca_dict["dmodel"]

    list_delete_key = []
    for k in nca_dict:
      if k[0] == "_":
        list_delete_key.append(k)

    for k in list_delete_key:
      del nca_dict[k]

    return nca_dict

if __name__ == "__main__":
  nca = CriticalNCA()
  nca.dmodel.summary()
