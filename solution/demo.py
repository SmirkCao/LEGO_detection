#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: demo
# Date: 4/15/19
# Author: 😏 <smirk dot cao at gmail dot com>
"""This is demo code for detection details.

Raises:
    NotImplementedError -- if sub class not implemented show method
TODO: save to video
TODO: get data from path
"""
from segmentation import SegAdaThresh
import cv2 as cv
import logging
import argparse


class Demo(object):
    def __init__(self, n_camera=0, store=False):
        self.cap = cv.VideoCapture(n_camera)
        self.store = store
        
    def show(self):
        raise NotImplementedError


class CircleFeatureDemo(Demo):
    def show(self):
        pass


class SegAdaThreshDemo(Demo):
    def show(self):
        seg = SegAdaThresh()
        while True:
            _, frame = self.cap.read()
            rst, objs = seg.run(frame)
            cv.imshow("SegAdaThreshDemo", rst)
            k = cv.waitKey(100)
            if k == ord("q"):
                break
        cv.destroyAllWindows()
        

if __name__ == '__main__':
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format_str)
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", required=False, help="type of demo")
    ap.add_argument("-c", "--camera_id", required=False, help="camera id")
    args = vars(ap.parse_args())

    demo_map = {"feas": CircleFeatureDemo,
                "seg_ada": SegAdaThreshDemo}
    
    demo = demo_map[args["type"]](n_camera=int(args["camera_id"]))
    demo.show()
    

else:
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format_str)
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", required=False, help="type of demo")
    ap.add_argument("-c", "--camera_id", required=False, help="camera id")
    args = vars(ap.parse_args())

    demo_map = {"feas": CircleFeatureDemo,
                "seg_ada": SegAdaThreshDemo}

    demo = demo_map[args["type"]](n_camera=int(args["camera_id"]))
    demo.show()
