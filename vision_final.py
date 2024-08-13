# -*- coding: utf-8 -*-
"""
Created on Wed June  1 12:13:27 2024

@author: jyshin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 Read
img = cv2.imread('C:/Users/jyshin/Desktop/vision/Model4.png', cv2.IMREAD_COLOR)
show_img = np.copy(img)

img_height, img_width = img.shape[:2]
y = int(img_height * 0.2)
x = int(img_width * 0.25)

w = int(img_width * 0.5)
h = int(img_height * 1.0)


labels = np.zeros(img.shape[:2], np.uint8)
# 그랩 컷 적용, 사각형을 기준으로 배경, 전경으로 분할
labels, bgdModel, fgdModel = cv2.grabCut(img, labels, (x,y,w,h), None, None, 5, cv2.GC_INIT_WITH_RECT)

show_img = np.copy(img)
# 배경은 어둡게 표현
show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] = 0

cv2.imshow('image',show_img)
cv2.waitKey()
cv2.destroyAllWindows()

image = show_img.astype(np.float32) / 255.
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB);  #컬러 공간 변환

data = image_lab.reshape((-1,3))

num_classes = 7 # k 값

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
# 픽셀을 num_classes 의 클러스터로 묶음
# 랜덤 센터 시작
_, labels, centers = cv2.kmeans(data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

segmented_lab = centers[labels.flatten()].reshape(image.shape)
segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2BGR)

cv2.imshow('segmented', segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()

img0 = cv2.imread('C:/Users/jyshin/Desktop/vision/case3.png', cv2.IMREAD_COLOR)
img1 = cv2.imread('C:/Users/jyshin/Desktop/vision/case4.png', cv2.IMREAD_COLOR)

# SIFT 객체 생성, 200개의 특징점 추출

detector = cv2.SIFT().create(200)


# ORB를 사용해 두 이미지의 특징점, 기술자 계산
kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)

# 두 이미지를 매칭 (SIFT)

matcher = cv2.BFMatcher().create(cv2.NORM_L2, False)
matches = matcher.match(fea0, fea1)

# RANSAC 알고리즘을 사용해 호모그래피 계산
# 호모그래피 : 두 평면 사이의 투시변환 관계
pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

dbg_img_all = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)
dbg_img_filtered = cv2.drawMatches(img0, kps0, img1, kps1, [m for i, m in enumerate(matches) if mask[i]], None)

resized_img1 = cv2.resize(dbg_img_all, (dbg_img_all.shape[1] // 2, dbg_img_all.shape[0] // 2))
resized_img2 = cv2.resize(dbg_img_filtered, (dbg_img_filtered.shape[1] // 2, dbg_img_filtered.shape[0] // 2))

cv2.imshow('All Matches', resized_img1)
cv2.imshow('Filtered Matches', resized_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()