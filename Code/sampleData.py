import cv2
import os
from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt

directory = r'/home/hanhdz/deepLearning/ProJect/FaceRecog_FaceNet/DataImage'
listPerson = os.listdir(directory)
for person in listPerson:
    folderPerson = directory + '/' + person
    sampleNumbers = np.random.randint(1, len(os.listdir(folderPerson)) + 1, size=50)
    remainNumbers = np.setdiff1d(np.arange(1, len(os.listdir(folderPerson)) + 1), sampleNumbers)
    for number in remainNumbers:
        path = folderPerson + '/' + person + str(number) + '.png'
        os.remove(path)