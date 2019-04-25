#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: feas
# Date: 4/15/19
# Author: 😏 <smirk dot cao at gmail dot com>
from segmentation import SegAdaThresh
import cv2 as cv
import numpy as np
import warnings
import logging
import argparse
import os


class Feature(object):
    def extract(self, img, mask=None):
        raise NotImplementedError
    
    def get_feature(self, img, mask=None):
        logger.info("Not Implemented Error")
        return None


class FeaCircle(Feature):
    """
    for studs detect
    """
    # circle count, max diameter
    def extract(self, img, mask=None):
        c = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=200, maxRadius=300)
        circles = c[0, :, :]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)
            cv.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)
        cv.imshow("test", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        feas = None
        return feas


class FeaColor(Feature):
    def __init__(self):
        self.color_map = {0: "yellow", 1: "green", 2: "blue",
                          3: "red", 4: "white", 5: "orange",
                          6: "darkgray"}
        self.color_table = []
        for idx in range(7):
            self.color_table.append(np.load("color_table/"+self.color_map[idx]+".npy"))
        # print(self.color_table)
        
    def extract(self, img, mask=None):
        contour = mask
        rotated_rect = cv.minAreaRect(contour)
        x, y = rotated_rect[0]
        w, h = rotated_rect[1]
        angle = rotated_rect[2]
        rot_mat = cv.getRotationMatrix2D((x, y), angle, 1)
        rot_img = cv.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
        obj = rot_img[int(y) - int(h / 2):int(y) + int(h / 2), int(x) - int(w / 2):int(x) + int(w / 2)]
        cv.imshow("img", obj)
        # cv.imshow("img", img[int(w-x):int(x), int(y):int(y+h)])
        hsv = cv.cvtColor(obj, cv.COLOR_BGR2HSV)
        hist = cv.calcHist(hsv, [0, 1], None, (8, 8), [0, 180, 0, 255])
        # hist = cv.normalize(hist, cv.NORM_MINMAX)
        rst = np.array([cv.matchTemplate(hist, color, cv.HISTCMP_CORREL) for color in self.color_table])
        # np.save("color.npy", hist)
        idx = int(np.argmin(rst))
        return self.color_map[idx]
    
    def get_feature(self, img, mask=None):
        color = self.extract(img, mask=mask)
        for idx in range(7):
            if self.color_map[idx] == color:
                break
        return np.array([[idx]])


class FeaPerimeter(Feature):
    def extract(self, img, mask=None):
        contour = mask
        perimeter = cv.arcLength(contour, True)
        return perimeter
    
    def get_feature(self, img, mask=None):
        perimeter = self.extract(img, mask=mask)
        return np.array([[perimeter]])
    

class FeaArea(Feature):
    def extract(self, img, mask=None):
        contour = mask
        area = cv.contourArea(contour)
        return area
    
    def get_feature(self, img, mask=None):
        area = self.extract(img, mask=mask)
        return np.array([[area]])


class FeaMinAreaRect(Feature):
    """
    return centorid and area
    centroid for grab application
    """
    def extract(self, img, mask=None):
        contour = mask
        rst = cv.minAreaRect(contour)
        x, y = rst[0]
        w, h = rst[1]
        return (x, y), w*h
    
    def get_feature(self, img, mask=None):
        _, min_area = self.extract(img, mask=mask)
        return np.array([[min_area]])
    

class FeaHuMoments(Feature):
    def extract(self, img, mask=None):
        pass
    
    def get_feature(self, img, mask=None):
        pass
    
    
class FeaGoodFeatures(Feature):
    def extract(self, img, mask=None):
        contour = mask
        rotated_rect = cv.minAreaRect(contour)
        x, y = rotated_rect[0]
        w, h = rotated_rect[1]
        angle = rotated_rect[2]
        rot_mat = cv.getRotationMatrix2D((x, y), angle, 1)
        rot_img = cv.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
        # obj = rot_img[int(y) - int(h / 2):int(y) + int(h / 2), int(x) - int(w / 2):int(x) + int(w / 2)]
        gray = cv.cvtColor(rot_img, cv.COLOR_BGR2GRAY)
        
        corners = cv.goodFeaturesToTrack(gray, 500, 0.01, 15)
        return corners
    
    def get_feature(self, img, mask=None):
        corners = self.extract(img, mask=mask)
        return np.array([[corners.shape[0]]])


class FeatureExtractor(object):
    """
    file structure:
    train: input_path/train/filename
    test: input_path/test/filename
    """
    
    def __init__(self, input_path=None, output_path=None, verbose=True):
        self.input_path = input_path
        self.output_path = output_path
        self.files = []
        self.verbose = verbose
        self.feas = None
        self.seg = SegAdaThresh(min_area=1000, min_perimeter=100)

    def load_images(self):
        # file path
        self.files = []
        for root, dirs, files in os.walk(self.input_path):
            for file in files:
                if file.__contains__(".png"):
                    self.files.append(root + "/" + file)
        if self.verbose:
            logger.info(self.files)
        return self.files
    
    def process(self):
        labels = []
        x = []
        for file in self.files:
            features = None
            img = cv.imread(file)
            rst, objs = self.seg.run(img.copy())
            if "contours" in objs.keys():
                contours = objs["contours"]
                if len(contours) > 0:
                    contour = contours[0]
                    file = file.replace(self.input_path, self.output_path)
                    
                    # get features
                    for fea in self.feas:
                        if features is None:
                            features = fea.get_feature(img, contour)
                        else:
                            features = np.concatenate([features, fea.get_feature(img, contour)], axis=1)
                    x.append(features)
                    # get labels
                    logger.info(file)
                    labels.append(file.split("/")[-1].split("_")[0])
                else:
                    logger.info(file+" feature extraction failed.")
                    # x.append(x[-1])
                
        # logger.info("images count: {0}".format(len(labels)))
        # add label
        x = np.array(x)
        x = np.concatenate([np.array(x).reshape(len(labels), -1), np.array(labels).reshape(-1, 1)], axis=1)
        print("x shape, labels shape", x.shape, np.array(labels).shape)
        
        if self.verbose:
            logger.info("x shape: {0}".format(x.shape))
        # output
        fpath = os.path.join(self.output_path, "fea.npy")
        if self.verbose:
            logger.info("output feature files {0}.".format(self.output_path))
        np.save(fpath, x)
    
    def do(self, image, mask=None):
        x = []
        contour = mask
        features = None
        for fea in self.feas:
            if features is None:
                features = fea.get_feature(image, contour)
            else:
                features = np.concatenate([features, fea.get_feature(image, contour)], axis=1)
        x.append(features)
        x = np.array(x).reshape(1, -1)
        logger.info(x.shape)
        return x
        
    def summary(self):
        pass
    
    def add(self, feature):
        # assert feature
        if self.feas is None:
            self.feas = [feature]
        else:
            self.feas.append(feature)
            print("new feature")
            

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    ap.add_argument("-i", "--input_path", required=False, help="path to input data file")
    ap.add_argument("-o", "--output_path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    
    logger.info("input path:" + args["input_path"])
    
    if args["input_path"] is None:
        input_path = "dataset"
    else:
        input_path = args["input_path"]
    
    if args["output_path"] is None:
        output_path = "feas"
    else:
        output_path = args["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    fe = FeatureExtractor(input_path, output_path)
    fe.add(FeaPerimeter())
    fe.add(FeaArea())
    fe.add(FeaMinAreaRect())
    fe.add(FeaGoodFeatures())
    fe.load_images()
    fe.process()
