#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  LEGO_detection
# Filename: segmentation
# Date: 4/15/19
# Author: 😏 <smirk dot cao at gmail dot com>
import cv2 as cv
import numpy as np


class Segmenter(object):
    def __init__(self, min_area=2000, min_perimeter=150):
        self.min_area = min_area
        self.min_perimeter = min_perimeter
        
    def run(self, img):
        objs = None
        return objs
    
    
class SegAdaThresh(Segmenter):
    """
    area and perimeter for filter contour
    """
    def run(self, img):
        adaptive_method = 0
        threshold_type = 1
        block_size = 5
        const_value = 4
        max_value = 200
        # -1 or np.ones*255
        img_info = np.zeros((img.shape[0], int(img.shape[1]/2), img.shape[2]), img.dtype)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.blur(gray, (2, 2))
        thresh = cv.adaptiveThreshold(blur, max_value, adaptive_method, threshold_type, block_size, const_value)
        _, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rst = dict()
        img_rst = img.copy()
        if len(contours) > 0:
            areas = []
            perimeters = []
            contours_filtered = []
            idx = 0
            for contour in contours:
                area = cv.contourArea(contour)
                perimeter = cv.arcLength(contour, True)
                if area > self.min_area and perimeter > self.min_perimeter:
                    areas.append(area)
                    perimeters.append(perimeter)
                    contours_filtered.append(contour)
                    # tag on img
                    cv.drawContours(img, contour, -1, (0, 0, 255), 3, cv.LINE_AA)
                    cv.putText(img, "#{}".format(idx), tuple(contour[0, 0]),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
                    
                    # tag on img_info
                    cv.putText(img_info, "Perimeter:#{0}:{1:.1f}".format(idx, perimeter), (10, 20+idx*40),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (51, 204, 153))
                    cv.putText(img_info, "Area:#{0}:{1:.1f}".format(idx, area), (10, 40 + idx * 40),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (51, 204, 153))
                    idx += 1

            rst["areas"] = areas
            rst["perimeters"] = perimeters
            rst["contours"] = contours_filtered

            img_rst = np.concatenate([img_info, img], axis=1)
        return img_rst, rst

