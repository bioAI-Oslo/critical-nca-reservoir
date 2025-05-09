"""
Test NCA for criticality.
"""
# import numpy as np
from critical_nca import CriticalNCA
import utils
import os
from evaluate_criticality import evaluate_nca

def test(args, gen, ckpt):
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
  # del args.nca_model["padding_size"]

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
  s = utils.get_flat_weights(nca.weights)
  fit, val_dict = evaluate_nca(s, args, test=gen)
  print("Fitness: ", fit)
  print("Info: ", val_dict)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", help="path to log directory")
  parser.add_argument("--ckpt", default="", help="path to log directory")
  parser.add_argument('--repeat', default=1, type=int)
  p_args = parser.parse_args()

  if p_args.logdir:
    args_filename = os.path.join(p_args.logdir, "args.json")
    argsio = utils.ArgsIO(args_filename)
    args = argsio.load_json()
    for i in range(p_args.repeat):
      print(i)
      test(args, i+1, p_args.ckpt)

  else:
    print("Add --logdir [path/to/log]")
