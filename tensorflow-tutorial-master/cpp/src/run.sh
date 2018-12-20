#!/bin/bash

folder_dir=/home/jwei/tensorflow_c/tensorflow-tutorial-master/cpp
model_path=${folder_dir}/model/nn_model_frozen.pb

#cp binary to root folder
#cp ./build/cpptensorflow ./cpptensorflow

./build/cpptensorflow ${model_path}
