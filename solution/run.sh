#!/usr/bin/env bash

# classification pipeline
# smirk.cao@gmail.com

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "train data: input/train/*.png"
echo "test data: input/test/*.png"
## classification
## 0. folders preparation
## 0.1 input
#train="input/train"
#test="input/test"
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
### 2. gscv train
#python model.py -i ${train_fea} -o ${model}
### 3. classification
### 3.1 feature extraction
#python feas.py -i ${test} -o ${test_fea}
### 3.3 predict
#python classification.py -i ${test_fea} -m ${model}"/model.pickle"

## demo
### -c for camera id
#### segmentation
#python solution/demo.py -t seg_ada -c 1
#### color
#python solution/demo.py -t color -c 1
#### perimeter
#python solution/demo.py -t perimeter -c 1
#### area
#python solution/demo.py -t area -c 1
#### min area rect
#python solution/demo.py -t min_area -c 1
## good features
#python solution/demo.py -t good_feas -c 1
## clf demo
python solution/demo.py -t clf -c 1