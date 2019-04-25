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
"""
from segmentation import SegAdaThresh
from feas import FeaColor, FeaPerimeter, FeaArea, FeaMinAreaRect, FeatureExtractor
from model import RF
import cv2 as cv
import logging
import argparse
import pickle

COMMENT_COLOR = (255, 0, 0)


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
            # print(objs)
            cv.imshow("SegAdaThreshDemo", rst)
            k = cv.waitKey(100)
            if k == ord("q"):
                break
        cv.destroyAllWindows()
      
        
class ColorDemo(Demo):
    def show(self):
        seg = SegAdaThresh()
        fea = FeaColor()
        while True:
            _, frame = self.cap.read()
            rst, objs = seg.run(frame.copy())
            colors = []
            if "contours" in objs.keys():
                contours = objs["contours"]
                for idx, contour in enumerate(contours):
                    color = fea.extract(frame, contour)
                    colors.append(color)
                    x, y = tuple(contour[0, 0])
                    cv.putText(rst, "#{0}:{1}".format(idx, color), (x+int(frame.shape[0]/2), y+20),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COMMENT_COLOR)
            
            cv.imshow("ColorDemo", rst)
            k = cv.waitKey(100)
            if k == ord("q"):
                break
        cv.destroyAllWindows()


class PerimeterDemo(Demo):
    def show(self):
        seg = SegAdaThresh()
        fea = FeaPerimeter()
        while True:
            _, frame = self.cap.read()
            rst, objs = seg.run(frame.copy())
            perimeters = []
            if "contours" in objs.keys():
                contours = objs["contours"]
                for idx, contour in enumerate(contours):
                    perimeter = fea.extract(frame, contour)
                    perimeters.append(perimeter)
                    x, y = tuple(contour[0, 0])
                    cv.putText(rst, "Perimeter:#{0}:{1:.1f}".format(idx, perimeter),
                               (x + int(frame.shape[0] / 2), y + 20),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COMMENT_COLOR)
    
            cv.imshow("PerimeterDemo", rst)
            k = cv.waitKey(100)
            if k == ord("q"):
                break
        cv.destroyAllWindows()


class AreaDemo(Demo):
    def show(self):
        seg = SegAdaThresh()
        fea = FeaArea()
        while True:
            _, frame = self.cap.read()
            rst, objs = seg.run(frame.copy())
            areas = []
            if "contours" in objs.keys():
                contours = objs["contours"]
                for idx, contour in enumerate(contours):
                    area = fea.extract(frame, contour)
                    areas.append(area)
                    x, y = tuple(contour[0, 0])
                    cv.putText(rst, "Area:#{0}:{1:.1f}".format(idx, area),
                               (x + int(frame.shape[0] / 2), y + 20),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COMMENT_COLOR)
            
            cv.imshow("AreaDemo", rst)
            k = cv.waitKey(100)
            if k == ord("q"):
                break
        cv.destroyAllWindows()


class MinAreaRectDemo(Demo):
    def show(self):
        seg = SegAdaThresh()
        fea = FeaMinAreaRect()
        while True:
            _, frame = self.cap.read()
            rst, objs = seg.run(frame.copy())
            areas = []
            if "contours" in objs.keys():
                contours = objs["contours"]
                for idx, contour in enumerate(contours):
                    (x, y), area = fea.extract(frame, contour)
                    x, y = int(x), int(y)
                    areas.append(area)
                    cv.circle(rst, (x+int(frame.shape[1]/2), y), 5, COMMENT_COLOR, 3, cv.LINE_AA)
                    cv.putText(rst, "minArea:#{0}:{1:.1f}".format(idx, area),
                               (x + int(frame.shape[0] / 2), y + 20),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COMMENT_COLOR)
            
            cv.imshow("MinAreaRectDemo", rst)
            k = cv.waitKey(100)
            if k == ord("q"):
                break
        cv.destroyAllWindows()


class ClassificationDemo(Demo):
    def predict(self):
        print(self.model)
        return y
    
    def show(self):
        with open("label.pickle", "rb") as f:
            labels = pickle.load(f)
        
        seg = SegAdaThresh(min_area=1000, min_perimeter=100)
        fe = FeatureExtractor()
        fe.add(FeaPerimeter())
        fe.add(FeaArea())
        fe.add(FeaMinAreaRect())
        with open("model.pickle", "rb") as f:
            model = pickle.load(f)
        while True:
            _, frame = self.cap.read()
            rst, objs = seg.run(frame.copy())
            areas = []
            if "contours" in objs.keys():
                contours = objs["contours"]
                for idx, contour in enumerate(contours):
                    feas = fe.do(frame, contour)
                    y = model.predict(feas)
                    pos_x, pos_y = tuple(contour[0, 0])
                    cv.putText(rst, str(y),
                               (pos_x+int(frame.shape[1]/2)-50, pos_y+20),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COMMENT_COLOR)
                    
                    rst[200+40*idx:200+40*(idx+1), 40:40+40] = labels[y[0]]
    
            cv.imshow("Classification Demo", rst)
            k = cv.waitKey(10)
            if k == ord("q"):
                break
        cv.destroyAllWindows()
        

format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=False, help="type of demo")
ap.add_argument("-c", "--camera_id", required=False, help="camera id")
args = vars(ap.parse_args())

demo_map = {"feas": CircleFeatureDemo,
            "seg_ada": SegAdaThreshDemo,
            "color": ColorDemo,
            "perimeter": PerimeterDemo,
            "area": AreaDemo,
            "min_area": MinAreaRectDemo,
            "clf": ClassificationDemo}

if __name__ == '__main__':
    demo = demo_map[args["type"]](n_camera=int(args["camera_id"]))
    demo.show()

