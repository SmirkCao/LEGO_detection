#!/usr/bin/env bash

# classification pipeline
# smirk.cao@gmail.com

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "train data: input/train/label/*.jpg"
echo "test data: input/test/0/*.jpg"

## 0. folders preparation
## 0.1 input
#train="input/train"
#test="input/test/0"
#label="input/dataset.desc"
## 0.2 output
#output_folder="output/"$(date "+%Y%m%d%H%M%S")
#train_fea=${output_folder}"/train_fea"
#test_fea=${output_folder}"/test_fea"
#model=${output_folder}"/model"
#detect=${output_folder}"/detect"
## 0.3 mkdir
#mkdir -p {${output_folder},${train_fea},${model},${detect},${test_fea}}
#
## 1. feature extraction
#python feas.py -i ${train} -o ${train_fea}
## 2. gscv train
#python model.py -i ${train_fea} -o ${model}
## 3. classification
## 3.1 detection
#python detect.py -i ${test} -o ${detect}
## 3.2 feature extraction
#python feas.py -i ${detect} -o ${test_fea}
## 3.3 predict
#python classification.py -i ${test_fea} -m ${model} -l ${label}

## demo
python solution/demo.py -t seg_ada -c 1
