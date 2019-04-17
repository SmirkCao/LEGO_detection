#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: utils
# Date: 4/16/19
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging
import cv2 as cv
import os
import pickle


class Dataset(object):
    def __init__(self, label, color, camera_id=1):
        self.label = label
        self.color = color
        self.cap = cv.VideoCapture(camera_id)
        
    def collect(self):
        idx = 0
        while True:
            _, frame = self.cap.read()
            cv.imshow("frame", frame)
            fname = self.label+"_"+self.color+"_"+"{0:0>3}".format(idx)+".png"
            cv.imwrite(fname, frame)
            k = cv.waitKey(1000)
            if k == ord("q"):
                break
            idx += 1
                
        cv.destroyAllWindows()
        self.cap.release()


class Label(object):
    def __init__(self, path):
        self.input_path = path
        
    def collect(self):
        rst = dict()
        for root, dirs, files in os.walk(self.input_path):
            for file in files:
                if file.__contains__(".png"):
                    # print(root+"/"+file, file, file[:-4])
                    img = cv.imread(root + "/" + file)
                    img = cv.resize(img, (40, 40))
                    rst[file[:-4]] = img
        with open("label.pickle", "wb") as f:
            pickle.dump(rst, f)
        
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False, help="path to input data file")
ap.add_argument("-l", "--label", required=False, help="label")
ap.add_argument("-c", "--color", required=False, help="color")
ap.add_argument("-t", "--type", required=False, help="type")

args = vars(ap.parse_args())

if __name__ == '__main__':
    if args["type"] == "dataset":
        dat = Dataset(args["label"], args["color"])
        dat.collect()
    if args["type"] == "label":
        label = Label(args["path"])
        label.collect()
