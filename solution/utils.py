#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: utils
# Date: 4/16/19
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging
import cv2 as cv


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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False, help="path to input data file")
ap.add_argument("-l", "--label", required=False, help="label")
ap.add_argument("-c", "--color", required=False, help="color")

args = vars(ap.parse_args())

if __name__ == '__main__':
    dat = Dataset(args["label"], args["color"])
    dat.collect()

else:
    pass