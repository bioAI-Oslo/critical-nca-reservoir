# Reservoir Computing with Evolved Critical Neural Cellular Automata
This repository contains the complete project for:

Reservoir Computing with Evolved Critical Neural Cellular Automata

Dependencies used:
* Python 3.10.8
* TensorFlow 2.10.1
* keras 2.10.0
* numpy 1.26.4
* scikit-learn 1.2.2
* powerlaw 1.5
* EvoDynamic (commit 83a15c8bb18ecb7da8cbc83ce6092d477aeae459)

NCA evolution towards criticality:
* train_nca.py
* test_nca.py

5-bit memory task:
* ReCA_X-bit_memory_NCA.py (Ours - use checkpoint in the folder)
* ReCA_X-bit_memory.py (Original)

MNIST classification:
* reservoir_mnist_make_dataset.py: Make dataset modified by NCA
* reservoir_mnist.py: Train and test of the reservoir (NCA)

