#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: segmentation
# Date: 4/15/19
# Author: 😏 <smirk dot cao at gmail dot com>
import cv2 as cv


class Segmenter(object):
    def run(self, img):
        objs = None
        return objs
    
    
class SegAdaThresh(Segmenter):
    def run(self, img):
        adaptive_method = 0
        threshold_type = 1
        block_size = 5
        const_value = 4
        max_value = 200
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.blur(gray, (2, 2))
        rst = cv.adaptiveThreshold(blur, max_value, adaptive_method, threshold_type, block_size, const_value)
        _, contours, _ = cv.findContours(rst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cv.drawContours(img, contours[0], -1, (0, 0, 255), 3)
        return img, contours

