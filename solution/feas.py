#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: feas
# Date: 4/15/19
# Author: 😏 <smirk dot cao at gmail dot com>
import cv2 as cv
import numpy as np


class Feature(object):
    def extract(self, img, mask=None):
        raise NotImplementedError


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
                          6: "DarkGray"}
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


class FeaPerimeter(Feature):
    def extract(self, img, mask=None):
        contour = mask
        perimeter = cv.arcLength(contour, True)
        return perimeter
    

class FeaArea(Feature):
    def extract(self, img, mask=None):
        contour = mask
        area = cv.contourArea(contour)
        return area
    

class FeaMinAreaRect(Feature):
    def extract(self, img, mask=None):
        contour = mask
        pass


class FeaHuMoments(Feature):
    def extract(self, img, mask=None):
        pass
