#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: classification
# Date: 4/17/19
# Author: 😏 <smirk dot cao at gmail dot com>
from model import RF
import argparse
import logging
import json
import pickle


class Demo(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.labels = None
    
    def load_model(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
    
    def predict(self):
        print(self.model)
        y = self.model.predict(self.model.x)
        return y


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    ap.add_argument("-i", "--input_path", required=False, help="path to input data file")
    ap.add_argument("-m", "--model_path", required=False, help="path to model file")
    args = vars(ap.parse_args())
    logger.info(args)
    
    if args["input_path"] is None:
        input_path = "dataset/"
    else:
        input_path = args["input_path"]
    
    model_path = args["model_path"]
    
    demo = Demo(model_path)
    demo.load_model()
    demo.model.load_data(input_path + "/fea.npy")
    rst = demo.model.predict()
    logger.info(rst)
    logger.info(demo.model.y)
