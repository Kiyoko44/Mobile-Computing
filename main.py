# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import csv
import numpy as np
import os
import tensorflow as tf
from skimage import io

## import the handfeature extractor class
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

# =============================================================================
# Other Arrays/Variables
# =============================================================================

dictionary = {'Num0':0,'Num1':1,'Num2':2,'Num3':3,
            'Num4':4,'Num5':5,'Num6':6,'Num7':7,
            'Num8':8,'Num9':9,'FanDown':10,
            'FanOn':11,'FanOff':12, 'FanUp':13,
            'LightOff':14,'LightOn':15,'SetThermo':16}

trainingArray = {'Num0':[],'Num1':[],'Num2':[],'Num3':[],
            'Num4':[],'Num5':[],'Num6':[],'Num7':[],
            'Num8':[],'Num9':[],'FanDown':[],
            'FanOn':[],'FanOff':[], 'FanUp':[],
            'LightOff':[],'LightOn':[],'SetThermo':[]}
trainArray = {}
trainPath = "./traindata"
trainFramePath = "./trainFrames"
testArray = []
testPath="./test"
testFramePath="./testFrames"
frame = HandShapeFeatureExtractor()
frame.get_instance()

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================

# Extract the middle frame for all training gesture videos
count = 0
for v in os.listdir(trainPath):
  print(v)
  frameExtractor(trainPath + "/" + v,trainFramePath,count)
  key = ""
  if count < 9:
    key = "0000" + str(count+1) + ".png"
  else:
    key = "000" + str(count+1) + ".png"
  trainArray[key] = v
  count+=1
for i in os.listdir(trainFramePath):
  iPath = "/content/trainFrames/" + i
  image = io.imread(iPath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imageArray = frame.extract_feature(gray)
  splits = trainArray[i].split("_")[0]
  trainingArray[splits].append(imageArray)
print("END")

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================

# Extract the middle frame for all test gesture videos
count = 0
for v in os.listdir(testPath):
  print(v)
  frameExtractor(testPath + "/" + v,testFramePath,count)
  count+=1
for i in os.listdir(testFramePath):
  iPath = "/content/testFrames/" + i
  image = io.imread(iPath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imageArray = frame.extract_feature(gray)
  testArray.append(imageArray)


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

MAX = 0
g = -1
results = []
for t in testArray:
  for k in trainingArray:
    loss = tf.keras.losses.CosineSimilarity(axis = 1)
    row1 = loss(trainingArray[k][0], t).numpy()
    row2 = loss(trainingArray[k][1], t).numpy()
    row3 = loss(trainingArray[k][2], t).numpy()
    avg = (row1 + row2 + row3)/3
    if avg > MAX or g == -1:
      MAX = avg
      g = dictionary[k]
  results.append(g)
  g = -1
print("Result Array: ")
print(results)
np.savetxt("Results.csv", results, fmt="%d")
 