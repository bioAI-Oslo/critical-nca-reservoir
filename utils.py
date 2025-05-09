"""
Utils
"""
import tensorflow as tf
import numpy as np
import json
from types import SimpleNamespace
import os
import csv

def get_weights_info(weights):
  weight_shape_list = []
  for layer in weights:
    weight_shape_list.append(tf.shape(layer))

  weight_amount_list = [tf.reduce_prod(w_shape)\
                        for w_shape in weight_shape_list]
  weight_amount = tf.reduce_sum(weight_amount_list)

  return weight_shape_list, weight_amount_list, weight_amount


def get_model_weights(flat_weights, weight_amount_list, weight_shape_list):
  split_weight = tf.split(flat_weights, weight_amount_list)
  return [tf.reshape(split_weight[i], weight_shape_list[i])\
          for i in tf.range(len(weight_shape_list))]

def get_flat_weights(weights):
  flat_weights = []
  for layer in weights:
    flat_weights.extend(list(layer.numpy().flatten()))
  return flat_weights


# Code from: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def fig2array(fig):
  # If not drawn
  fig.canvas.draw()
  # Now we can save it to a numpy array.
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data

# Based on: https://stackoverflow.com/questions/57541522/save-load-a-dictionary
class ArgsIO:
  def __init__(self, filename):
    self.filename = filename

  def save_json(self):
    print(self.__dict__)
    with open(self.filename, 'w') as f:
      f.write(json.dumps(self.__dict__))

  def load_json(self):
    with open(self.filename, 'r') as f:
      dictionary = json.loads(f.read())
    return SimpleNamespace(**dictionary)


def save_generation(loss, features, gen, folder_path):
  csv_file_path = os.path.join(folder_path, "{:06d}.csv".format(gen))
  with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["features", "loss",
                     "solution"])
    for i in range(len(loss)):
      writer.writerow([features[i],
                      loss[i]])

def save_generation_with_solutions(solutions, loss, features, gen, folder_path):
  csv_file_path = os.path.join(folder_path, "{:06d}.csv".format(gen))
  with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["features", "loss",
                     "solution"])
    for i in range(len(loss)):
      writer.writerow([features[i],
                      loss[i],
                      list(solutions[i])])

def reorganize_obs(body_grid, sensor_n, obs):
  obs_array = np.asarray(obs)
  new_obs = np.array(obs)
  flat_bg = body_grid.flatten()
  idx_array = np.arange(np.prod(body_grid.shape))
  for batch_idx in range(len(obs)):
    sensor_number_per_type = []
    idx_array_per_type = []
    for i in range(2,sensor_n+2):
      idx_array_i = idx_array[flat_bg==i]
      sensor_number_per_type.append(len(idx_array_i))
      idx_array_per_type.append(idx_array_i)

    idx_array_concat = np.concatenate(idx_array_per_type, 0)
    idx_array_sort_idx = np.argsort(idx_array_concat)
    new_obs[batch_idx] = obs_array[batch_idx, idx_array_sort_idx]

  return new_obs
