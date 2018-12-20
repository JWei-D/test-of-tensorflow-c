#!/bin/bash

#freeze the graph and the weights
python freeze_graph.py --input_graph=/home/jwei/tensorflow_c/codes/model/linermodel.pbtxt --input_checkpoint=/home/jwei/tensorflow_c/codes/model/linermodel.ckpt --output_graph=/home/jwei/tensorflow_c/codes/model/liner_freeze.pb --output_node_names=outputs
