#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: model
# Date: 4/17/19
# Author: 😏 <smirk dot cao at gmail dot com>
# Extract features from dataset to feas folder.
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
import numpy as np
import argparse
import logging
import os
import warnings
import pickle


class Model(object):
    def __init__(self, input_path, output_path):
        self.data = None
        self.x = None
        self.y = None
        self.input_path = input_path
        self.output_path = output_path
        self.clf = None
    
    def load_data(self, feas):
        self.data = np.load(feas)
        np.random.seed(2018)
        np.random.shuffle(self.data)
        print("data shape", self.data.shape)
        self.x = self.data[:, :-1]
        self.y = self.data[:, -1]
        print(self.data.shape)
    
    def fit(self, ):
        pass
    
    def predict(self, x=None):
        y = None
        return y
    
    def dump_model(self):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        fpath = os.path.join(output_path, "model.pickle")
        with open(fpath, "wb") as f:
            pickle.dump(self, f)
    
    def load_model(self):
        fpath = os.path.join(output_path, "model.pickle")
        with open(fpath, "rb") as f:
            self.clf = pickle.load(f)


class RF(Model):
    def __init__(self, input_path, output_path):
        super().__init__(input_path, output_path)
        self.clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    def fit(self):
        print(self.x.shape)
        print(self.y.shape, self.y)
        self.clf = self.clf.fit(self.x, self.y)
        scores = cross_val_score(self.clf, self.x, self.y, cv=5)
        logger.info(np.mean(scores))
    
    def predict(self, x=None):
        if x is None:
            x = self.x
        
        y = self.clf.predict(x)
        return y


class KNN(Model):
    def __init__(self, input_path, output_path):
        super().__init__(input_path, output_path)
        self.clf = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
    
    def fit(self):
        print(self.x.shape)
        print(self.y.shape, self.y)
        self.clf = self.clf.fit(self.x)

    def predict(self, x=None):
        if x is None:
            x = self.x
        
        y = self.clf.kneighbors(x)
        print(y)
        return y


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    ap.add_argument("-i", "--input_path", required=False, help="path to input data file")
    ap.add_argument("-o", "--output_path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    
    logger.info("input path:" + args["input_path"])
    
    if args["input_path"] is None:
        input_path = "dataset/"
    else:
        input_path = args["input_path"]
    
    if args["output_path"] is None:
        output_path = "feas"
    else:
        output_path = args["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    clf_rf = RF(input_path, output_path)
    data_path = os.path.join(input_path, "fea.npy")
    clf_rf.load_data(data_path)
    clf_rf.fit()
    clf_rf.dump_model()