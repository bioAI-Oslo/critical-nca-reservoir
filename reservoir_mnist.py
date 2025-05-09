import os
import tensorflow as tf
import numpy as np
from critical_nca import CriticalNCA
import utils
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import csv

plt.rcParams.update({'font.size': 14})


def get_nca(args, ckpt=""):
  print("Testing checkpoint saved in: " + args.log_dir)

  del args.nca_model["built"]
  del args.nca_model["inputs"]
  del args.nca_model["outputs"]
  del args.nca_model["input_names"]
  del args.nca_model["output_names"]
  del args.nca_model["stop_training"]
  del args.nca_model["history"]
  del args.nca_model["compiled_loss"]
  del args.nca_model["compiled_metrics"]
  del args.nca_model["optimizer"]
  del args.nca_model["train_function"]
  del args.nca_model["test_function"]
  del args.nca_model["predict_function"]
  del args.nca_model["channel_n"]

  nca = CriticalNCA(**args.nca_model)
  nca.dmodel.summary()

  ckpt_filename = ""
  if ckpt == "":
    checkpoint_filename = "checkpoint"
    with open(os.path.join(args.log_dir, checkpoint_filename), "r") as f:
      first_line = f.readline()
      start_idx = first_line.find(": ")
      ckpt_filename = first_line[start_idx+3:-2]
  else:
    ckpt_filename = os.path.basename(ckpt)

  print("Testing model with lowest training loss...")
  nca.load_weights(os.path.join(args.log_dir, ckpt_filename))

  return nca


def get_nca_output(nca, img, width, timesteps, img_num_pixel):
  x = np.zeros((1, width, nca.channel_n),
               dtype=np.float32)

  x[0, width//2, 0] = img[0]

  x_history = [x[0,:,0]]
  for t in range(timesteps-1):
    x = nca(x).numpy()
    x_history.append(x[0,:,0])
    if t < img_num_pixel:
      x[0, width//2, 0] = img[t]

  x_history_arr = np.array(x_history)

  return x_history_arr

def train_readout(args):
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = ((x_train / 255.0) > 0.5).astype(np.float64)
  x_train = x_train.reshape(x_train.shape[0],-1)


  x_test = ((x_test / 255.0) > 0.5).astype(np.float64)
  x_test = x_test.reshape(x_test.shape[0],-1)

  model = svm.LinearSVC()
  # model = MLPClassifier(hidden_layer_sizes=(), verbose=True)

  model.fit(x_train, np.squeeze(y_train))

  y_pred = model.predict(x_test)

  print(accuracy_score(np.squeeze(y_test), y_pred))


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", help="path to log directory")
  p_args = parser.parse_args()

  if p_args.logdir:
    args_filename = os.path.join(p_args.logdir, "args.json")
    argsio = utils.ArgsIO(args_filename)
    args = argsio.load_json()

    train_readout(args)

  else:
    print("Add --logdir [path/to/log]")
