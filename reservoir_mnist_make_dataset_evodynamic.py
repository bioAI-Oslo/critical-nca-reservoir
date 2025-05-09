import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np


import utils
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import csv

plt.rcParams.update({'font.size': 14})


def create_mnist_ca(img, width, timesteps):

  exp = experiment.Experiment()
  g_ca = exp.add_group_cells(name="g_ca", amount=width)
  neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
  g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init=img.reshape(g_ca.amount_with_batch).astype(np.float64))
  g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                             neighbors=neighbors,\
                                             center_idx=center_idx)

  fargs_list = [(a,) for a in [94]]

  exp.add_connection("g_ca_conn",
                     connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                   act.rule_binary_ca_1d_width3_func,
                                                   g_ca_bin_conn, fargs_list=fargs_list))

  exp.add_monitor("g_ca", "g_ca_bin")
  exp.initialize_cells()

  exp.run(timesteps=timesteps)

  ca_result = exp.get_monitor("g_ca", "g_ca_bin")
  print("ca_result.shape", ca_result.shape)

  return ca_result.reshape(-1)



def train_readout():
  width = 28*28
  timesteps = 4

  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = ((x_train / 255.0) > 0.5).astype(np.float64)
  x_train = x_train.reshape(x_train.shape[0],-1)


  x_test = ((x_test / 255.0) > 0.5).astype(np.float64)
  x_test = x_test.reshape(x_test.shape[0],-1)

  img_num_pixel = x_train.shape[1]

  with open('mnist_x_train_rule94.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i, img in enumerate(x_train):
      if i > 10:
        break
      print(i)

      nca_output_arr = create_mnist_ca(img, width, timesteps).astype(np.uint8)
      writer.writerow(nca_output_arr)

  with open('mnist_x_test_rule94.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i, img in enumerate(x_test):
      if i > 10:
        break
      print(i)

      nca_output_arr = create_mnist_ca(img, width, timesteps).astype(np.uint8)
      writer.writerow(nca_output_arr)

if __name__ == "__main__":
  train_readout()
